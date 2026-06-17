#!/usr/bin/env python3
"""
构造 SFT 训练数据集
用 MiMo-2.5 (OpenAI 兼容 API) 对 FVQA 训练集生成第一轮 response
输出: data/FVQA/sft_train.parquet

运行: python scripts/data_pipeline/05_construct_sft_data.py
"""

import os
import re
import sys
import json
import time
import base64
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image
from openai import OpenAI
from tqdm import tqdm

# ====== 配置 ======
BASE_DIR = os.path.join(os.path.dirname(__file__), '../..')
INPUT_PATH = os.path.join(BASE_DIR, 'data/FVQA/fvqa_train.parquet')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data/FVQA/sft_train.parquet')
REPORT_PATH = os.path.join(BASE_DIR, 'data/FVQA/sft_construction_report.json')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'data/FVQA/sft_checkpoint.jsonl')  # 增量保存

MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
MIMO_MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.5")

MAX_WORKERS = 4       # 串行处理，避免限流
MAX_RETRIES = 2
TEMPERATURE = 0.3
RATE_LIMIT_SLEEP = 0.5  # 每次请求间隔秒数
TARGET_PER_ACTION = 500  # 每种 action 的目标数量

# ====== Prompt (按 category 分) ======
SYSTEM_PROMPT_SEARCH_FREE = """You are given an image and a question. You can answer this question using your own knowledge without any external search.

Output exactly in this format:
<reason>your reasoning about the image and question</reason><answer>concise answer</answer>

Rules:
- You must include <reason>...</reason> before the answer
- The answer must be concise (1-5 words)
- Do NOT invoke any search tools"""

SYSTEM_PROMPT_SEARCH_REQUIRED = """You are given an image and a question. You CANNOT answer this question using your own knowledge — you MUST use a search tool.

You have two search options:
1. If you cannot identify the visual content in the image, output:
<reason>your reasoning about why you need image search</reason><search><img></search>

2. If you can identify the visual content but need factual knowledge about it, output:
<reason>your reasoning about what you need to search</reason><text_search>your search query</text_search>

Rules:
- You MUST use a search tool (option 1 or 2) — do NOT try to answer directly
- You must include <reason>...</reason> before the search tag
- Generate a specific, well-crafted search query"""


def image_to_base64(img_data):
    """Convert image data to base64 string. Handles FVQA format: np.ndarray of dict({'bytes': ...})."""
    if isinstance(img_data, np.ndarray):
        # FVQA format: numpy array of dicts with 'bytes' key
        first = img_data.flat[0]
        if isinstance(first, dict) and 'bytes' in first:
            img = Image.open(BytesIO(first['bytes']))
        else:
            img = Image.fromarray(img_data)
    elif isinstance(img_data, dict) and 'bytes' in img_data:
        img = Image.open(BytesIO(img_data['bytes']))
    else:
        img = Image.open(BytesIO(img_data))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    buf = BytesIO()
    img.save(buf, format='JPEG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def extract_question(prompt):
    """Extract question text from prompt format (handles numpy array and list)."""
    import numpy as np
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, list) and len(prompt) > 0:
        item = prompt[0]
        if isinstance(item, dict):
            return item.get('content', '')
        return str(item)
    return str(prompt)


def call_mimo(client, question, image_b64, system_prompt=None, retries=MAX_RETRIES):
    """Call MiMo API with image + question."""
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT_SEARCH_FREE
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MIMO_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": f"Question: {question}"}
                    ]}
                ],
                max_tokens=256,
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if '429' in str(e):
                wait = min(10, 2 ** attempt * 2)
                time.sleep(wait)
                continue
            if attempt < retries:
                time.sleep(1)
                continue
            print(f"  API error after {retries} retries: {e}")
            return None


def validate_format(response):
    """Check if response matches expected format."""
    if response is None:
        return False
    patterns = [
        r'^<reason>.*</reason>.*<answer>.*</answer>$',
        r'^<reason>.*</reason>.*<search><img></search>$',
        r'^<reason>.*</reason>.*<text_search>.*</text_search>$',
    ]
    return any(re.search(p, response, re.DOTALL) for p in patterns)


def check_category_match(response, category):
    """检查 response 是否匹配 category 要求."""
    if category == 'search_free':
        # search_free 不应该触发搜索
        if '<search>' in response or '<text_search>' in response:
            return False
        if '<answer>' not in response:
            return False
    else:
        # search_required 必须触发搜索，不能直接回答
        if '<answer>' in response and '<search>' not in response and '<text_search>' not in response:
            return False
        if '<search>' not in response and '<text_search>' not in response:
            return False
    return True


def check_ground_truth(response, category, gt_info):
    """检查 search_free 的答案是否和 ground_truth 匹配."""
    if category != 'search_free':
        return True  # search_required 不检查答案

    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if not answer_match:
        return False

    model_answer = answer_match.group(1).strip().lower()
    gt = gt_info.get('ground_truth', '').lower()
    cands_str = gt_info.get('candidate_answers', '')

    all_answers = [gt]
    if cands_str and cands_str != '[]':
        try:
            cands = eval(cands_str)
            all_answers.extend([c.lower() for c in cands])
        except:
            pass

    for a in all_answers:
        if a and (model_answer == a or model_answer in a or a in model_answer):
            return True
    return False


def process_sample(idx, row, client):
    """Process a single sample: call MiMo and validate."""
    question = extract_question(row['prompt'])
    image_b64 = image_to_base64(row['images'])
    category = row.get('category', 'unknown')

    # 按 category 选 prompt
    if category == 'search_free':
        system_prompt = SYSTEM_PROMPT_SEARCH_FREE
    else:
        system_prompt = SYSTEM_PROMPT_SEARCH_REQUIRED

    response = call_mimo(client, question, image_b64, system_prompt=system_prompt)

    if response is None:
        return None

    if not validate_format(response):
        return None

    if not check_category_match(response, category):
        return None

    # ground truth 过滤: search_free 答错的丢弃
    rm = row.get('reward_model', {})
    gt_info = rm if isinstance(rm, dict) else {}
    if not check_ground_truth(response, category, gt_info):
        return None

    return {
        'idx': idx,
        'data_id': row.get('data_id', f'fvqa_train_{idx}'),
        'category': category,
        'question': question,
        'response': response,
        'ground_truth': row['reward_model']['ground_truth'],
    }


def load_checkpoint():
    """Load already-processed results from checkpoint file. Key by data_id."""
    done_ids = set()
    results = []
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done_ids.add(r['data_id'])
                    results.append(r)
        print(f"Resumed {len(results)} samples from checkpoint")
    return done_ids, results


def save_checkpoint(result):
    """Append a single result to checkpoint file."""
    with open(CHECKPOINT_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


IMAGE_SEARCH_PATTERNS = [
    r'what is this',
    r'who is this',
    r'where is this',
    r'what (kind|type|brand) of',
    r'name (this|the)',
    r'identify',
    r'recognize',
    r'what .*(?:building|landmark|monument|statue|plant|animal|food|object|bridge|tower|temple)',
    r'which .*(?:building|landmark|city|country|place)',
    r'(?:painting|artwork|picture|photo|image) .*(?:show|depict|is)',
]


def needs_image_search(question):
    """简单规则判断问题是否大概率需要搜图."""
    q = question.lower()
    return any(re.search(p, q) for p in IMAGE_SEARCH_PATTERNS)


def get_action_type(response):
    """从 response 中提取 action 类型."""
    if '<answer>' in response:
        return 'direct_answer'
    elif '<search>' in response:
        return 'image_search'
    elif '<text_search>' in response:
        return 'text_search'
    return 'unknown'


def count_actions(results):
    """统计各 action 类型数量."""
    counts = {'direct_answer': 0, 'image_search': 0, 'text_search': 0}
    for r in results:
        action = get_action_type(r['response'])
        if action in counts:
            counts[action] += 1
    return counts


def is_target_reached(action_counts, target):
    """检查是否所有 action 都达到目标."""
    return all(v >= target for v in action_counts.values())


def main():
    print("=" * 60)
    print("SFT Data Construction (action-balanced)")
    print("=" * 60)

    # Load full data (no sampling)
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df)} samples from {INPUT_PATH}")

    if 'category' in df.columns:
        print(f"Category distribution:")
        print(df['category'].value_counts().to_string())

    # Resume from checkpoint
    done_ids, results = load_checkpoint()

    # 过滤旧 checkpoint 中 ground truth 不匹配的 search_free
    if len(results) > 0:
        orig_count = len(results)
        filtered_results = []
        for r in results:
            if r.get('category') == 'search_free':
                rm_row = df[df['data_id'] == r['data_id']]
                if len(rm_row) > 0:
                    rm = rm_row.iloc[0].get('reward_model', {})
                    gt_info = rm if isinstance(rm, dict) else {}
                    if not check_ground_truth(r['response'], 'search_free', gt_info):
                        done_ids.discard(r['data_id'])
                        continue
            filtered_results.append(r)
        if len(filtered_results) < orig_count:
            print(f"Filtered {orig_count - len(filtered_results)} wrong ground truth from checkpoint")
            # 重写 checkpoint
            with open(CHECKPOINT_PATH, 'w', encoding='utf-8') as f:
                for r in filtered_results:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
        results = filtered_results

    action_counts = count_actions(results)
    print(f"\nAlready done: {len(done_ids)}")
    print(f"Current action counts: {action_counts}")
    print(f"Target per action: {TARGET_PER_ACTION}")
    print(f"Need: direct={max(0, TARGET_PER_ACTION-action_counts['direct_answer'])}, "
          f"image={max(0, TARGET_PER_ACTION-action_counts['image_search'])}, "
          f"text={max(0, TARGET_PER_ACTION-action_counts['text_search'])}")

    success_new = 0
    failed_new = 0
    elapsed = 0

    if is_target_reached(action_counts, TARGET_PER_ACTION):
        print("\nAll targets reached! Skipping API calls.")
    else:
        # Init client
        client = OpenAI(api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL)
        print(f"\nUsing model: {MIMO_MODEL}")
        print(f"API base: {MIMO_BASE_URL}")
        print(f"Rate limit: {RATE_LIMIT_SLEEP}s per request")

        # 策略: search_free → direct_answer, search_required → image/text_search
        # 当 direct_answer 满了，跳过 search_free
        # 当 image_search 和 text_search 都满了，跳过 search_required
        remaining_df = df[~df['data_id'].isin(done_ids)]
        t0 = time.time()
        success_new = 0
        failed_new = 0

        for idx, row in tqdm(remaining_df.iterrows(), total=len(remaining_df), desc="Constructing SFT"):
            category = row.get('category', 'unknown')

            # 检查是否可以跳过
            if category == 'search_free' and action_counts['direct_answer'] >= TARGET_PER_ACTION:
                continue  # direct_answer 已满，跳过 search_free
            if category == 'search_required':
                image_full = action_counts['image_search'] >= TARGET_PER_ACTION
                text_full = action_counts['text_search'] >= TARGET_PER_ACTION
                if image_full and text_full:
                    continue  # 都满了，跳过
                # 只剩 image_search 没满时，预筛需要搜图的问题
                if text_full and not image_full:
                    question = extract_question(row['prompt'])
                    if not needs_image_search(question):
                        continue  # 不太可能需要搜图，跳过

            # 全部达成
            if is_target_reached(action_counts, TARGET_PER_ACTION):
                print(f"\nAll targets reached! Stopping early.")
                break

            result = process_sample(idx, row, client)

            if result is not None:
                results.append(result)
                save_checkpoint(result)
                action = get_action_type(result['response'])
                action_counts[action] = action_counts.get(action, 0) + 1
                success_new += 1
            else:
                failed_new += 1

            # Rate limiting
            time.sleep(RATE_LIMIT_SLEEP)

        elapsed = time.time() - t0

    # 最终统计
    action_counts = count_actions(results)
    success_categories = {}
    for r in results:
        cat = r['category']
        success_categories[cat] = success_categories.get(cat, 0) + 1

    # Trim to target: 每种 action 最多保留 TARGET_PER_ACTION
    final_results = []
    action_seen = {'direct_answer': 0, 'image_search': 0, 'text_search': 0}
    for r in results:
        action = get_action_type(r['response'])
        if action_seen.get(action, 0) < TARGET_PER_ACTION:
            final_results.append(r)
            action_seen[action] = action_seen.get(action, 0) + 1

    # Build SFT parquet from final results
    sft_rows = []
    for r in final_results:
        row = df.iloc[r['idx']]
        sft_rows.append({
            'messages': [
                {'content': f"<image>\n{r['question']}", 'role': 'user'},
                {'content': r['response'], 'role': 'assistant'},
            ],
            'images': row['images'],
            'data_source': 'mmsearch_r1/sft',
            'data_id': r['data_id'],
            'category': r['category'],
        })

    sft_df = pd.DataFrame(sft_rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    sft_df.to_parquet(OUTPUT_PATH, index=False)

    final_action_counts = count_actions(final_results)
    total_processed = success_new + failed_new

    report = {
        'input_samples': total_processed,
        'output_samples': len(final_results),
        'target_per_action': TARGET_PER_ACTION,
        'action_distribution': final_action_counts,
        'category_success': success_categories,
        'time_seconds': round(elapsed, 1),
        'throughput': f"{total_processed/elapsed:.1f}/s" if elapsed > 0 else "N/A",
        'model': MIMO_MODEL,
        'temperature': TEMPERATURE,
        'output_path': OUTPUT_PATH,
        'checkpoint_path': CHECKPOINT_PATH,
    }

    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("SFT Construction Report")
    print(f"{'=' * 60}")
    print(f"This run: {success_new} success | {failed_new} failed")
    print(f"Total:    {len(final_results)} samples")
    print(f"Target:   {TARGET_PER_ACTION} per action")
    print(f"Time:     {elapsed:.1f}s")
    print(f"\nAction distribution (final):")
    for action, count in final_action_counts.items():
        print(f"  {action}: {count}/{TARGET_PER_ACTION}")
    print(f"\nCategory breakdown:")
    for cat, count in sorted(success_categories.items()):
        print(f"  {cat}: {count}")
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == '__main__':
    main()
