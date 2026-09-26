from __future__ import annotations

import os
from typing import Any

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
def run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    html_str = html_bytes.decode(encoding)
    text = extract_plain_text(html_str)
    return text
    # raise NotImplementedError

import fasttext
model = fasttext.load_model("../lid.176.bin")
def run_identify_language(text: str) -> tuple[Any, float]:
    text = text.replace("\n", " ") # 换行会报错
    label_list, score_list = model.predict(text)
    type = label_list[0].replace("__label__", "")
    score = float(score_list[0])
    return (type, score)
    # raise NotImplementedError

import re
def run_mask_emails(text: str) -> tuple[str, int]:
    pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.subn(pattern, "|||EMAIL_ADDRESS|||", text) # subn这个函数完美符合返回值要求
    # raise NotImplementedError


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    pattern = r'(?:\+?1[-. \t]?)?(?:\(?\d{3}\)?[-. \t]?)?\d{3}[-. \t]?\d{4}'
    return re.subn(pattern, "|||PHONE_NUMBER|||", text)  # subn这个函数完美符合返回值要求
    # raise NotImplementedError


def run_mask_ips(text: str) -> tuple[str, int]:
    pattern_ipv4 = r'\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
    pattern_ipv6 = r'\b(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?::[0-9a-fA-F]{1,4}){1,6}|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]+|::(?:ffff(?::0{1,4})?:)?(?:(?:25[0-5]|(?:2[0-4]|1?[0-9])?[0-9])\.){3}(?:25[0-5]|(?:2[0-4]|1?[0-9])?[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1?[0-9])?[0-9])\.){3}(?:25[0-5]|(?:2[0-4]|1?[0-9])?[0-9]))\b'
    pattern = f"{pattern_ipv4}|{pattern_ipv6}"
    return re.subn(pattern, "|||IP_ADDRESS|||", text)
    # raise NotImplementedError


model_nsfw = fasttext.load_model("../jigsaw_fasttext_bigrams_nsfw_final.bin")
def run_classify_nsfw(text: str) -> tuple[Any, float]:
    text = text.replace("\n", " ")  # 换行会报错
    label_list, score_list = model_nsfw.predict(text)
    type = label_list[0].replace("__label__", "")
    score = float(score_list[0])
    return (type, score)
    # raise NotImplementedError


model_toxic = fasttext.load_model("../jigsaw_fasttext_bigrams_hatespeech_final.bin")
def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    text = text.replace("\n", " ")  # 换行会报错
    label_list, score_list = model_toxic.predict(text)
    type = label_list[0].replace("__label__", "")
    score = float(score_list[0])
    return (type, score)
    # raise NotImplementedError

from nltk.tokenize import word_tokenize
def run_gopher_quality_filter(text: str) -> bool:
    # nltk.download("punkt_tab", quiet=True)
    words = word_tokenize(text)
    words_count = len(words)

    # 包含少于50个或超过100,000个单词。
    if words_count < 50 or words_count > 100000:
        return False

    # 平均单词长度超出3到10个字符的范围。
    words_char_lenth = 0
    for word in words:
        words_char_lenth += len(word)
    if float(words_char_lenth / words_count) < 3 or float(words_char_lenth / words_count) > 10:
        return False

    # 超过30%的行以省略号（“...”）结尾。
    lines = text.splitlines()
    if lines:
        pattern = r'(\.\.\.|…)\s*$'
        ellipsis_count = sum(1 for line in lines if re.search(pattern, line))
        ratio = float(ellipsis_count / len(lines))
        if ratio > 0.3:
            return False

    # 包含至少一个字母字符的单词少于80%。
    high_quality_word_count = 0
    for word in words:
        for c in word:
            if c.isascii() and c.isalpha():
                high_quality_word_count += 1
                break
    if float(high_quality_word_count / words_count) < 0.8:
        return False

    return True
    # raise NotImplementedError

model_quality = fasttext.load_model("./quality_classifier.bin")
def run_classify_quality(text: str) -> tuple[Any, float]:
    text = text.strip()
    text = text.replace("\n", " ")
    label, score = model_quality.predict(text)
    # print(label, score)
    return label[0], score[0]
    # raise NotImplementedError

import hashlib
import os
from pathlib import Path
# 测试的逻辑是只要有重复的行就全部删除，而不是像下面minhash那样留下第一次出现的行
# 所以遍历两遍，第一遍统计第二遍删除
def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 用 dict，方便统计数量
    seen_hashes = {}

    for path in input_files:
        in_path = Path(path)

        with open(in_path, "r", encoding="utf-8") as f_in:
            for line in f_in:  # 逐行读取，防止大文件爆内存
                line_hash = hashlib.md5(line.encode("utf-8")).hexdigest()
                if seen_hashes.get(line_hash) is None:
                    seen_hashes[line_hash] = 1
                else:
                    seen_hashes[line_hash] += 1

    # 为每一个输入文件，在输出目录下创建一个同名文件
    for path in input_files:
        in_path = Path(path)
        out_path = output_dir / in_path.name
        with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                line_hash = hashlib.md5(line.encode("utf-8")).hexdigest()
                if seen_hashes[line_hash] == 1:
                    f_out.write(line)  # 原样写入，保留原格式

    # raise NotImplementedError

from nltk.tokenize import word_tokenize
import zlib
def compare_samilarity(
    input_minhash_list1: list[int], input_minhash_list2: list[int], jaccard_threshold: float
) -> bool:
    length = len(input_minhash_list1)
    jaccard = 0
    for i in range (length):
        if input_minhash_list1[i] == input_minhash_list2[i]:
            jaccard += 1

    return float(jaccard / length) >= jaccard_threshold
    # raise NotImplementedError


def run_minhash_deduplication(
        input_files: list[os.PathLike], # 待去重的原始文件路径列表
        num_hashes: int,                # MinHash 签名的长度 k (比如 100)
        num_bands: int,                 # LSH 的 Band 数量 b (比如 20)
        ngrams: int,                    # 提取词组的长度 n (比如 5-gram), 注意ngram使用滑动窗口获取，每次移动一格会有重叠
        jaccard_threshold: float,       # 判定为重复文档的相似度阈值 (比如 0.8)
        output_directory: os.PathLike,  # 保留下来的非重复文档存储目录
) -> None:
    files_minhash_list = []
    for path in input_files:
        with open(path, "r", encoding="utf-8") as f_in:
            text = f_in.read()
            words = word_tokenize(text)

            #若总词数不足就填0处理
            if len(words) < ngrams:
                files_minhash_list.append([0.0] * num_hashes)
                continue

            minhash_list = [float("inf") for _ in range(num_hashes)]
            for i in range(len(words) - ngrams + 1):
                word_congregate = " ".join(words[i: i + ngrams])
                for seed in range(num_hashes): # 需要num_hashes个hash函数
                    hash_val = zlib.crc32(word_congregate.encode('utf-8'), seed)
                    if minhash_list[seed] > hash_val:
                        minhash_list[seed] = hash_val

            files_minhash_list.append(minhash_list)

    r = num_hashes // num_bands # r是每个桶里面有多少值
    # 对于每个band
    bands = {} # 全局字典，key为每个file的minhash_list切分块 value为files id的列表
    for i in range(num_bands):
        # 要遍历一遍所有的文本
        for file_id, minhash_list in enumerate(files_minhash_list):
            minhash_band = tuple(minhash_list[i * r: (i + 1) * r]) + (i,)
            bands.setdefault(minhash_band, []).append(file_id)

    # 在全局维护一个 duplicates_to_remove = set()，用于专门记录需要剔除的文档,用于防止一删一留
    duplicates_to_remove = set() # 全局集合，存放删除的元素避免重复删除
    # 遍历桶，判定相似度，寻找相似簇
    for band in bands.values():
        if len(band) >= 2:
            for i in range(len(band)):
                if band[i] not in duplicates_to_remove:
                    id1 = band[i]
                    for j in range(i+1, len(band)):
                        id2 = band[j]
                        if compare_samilarity(files_minhash_list[id1], files_minhash_list[id2], jaccard_threshold):
                            # 仅将id大的文件删除，所以只需判断id2是否在删除集合中就可以了。一个相似簇只需要一个文件
                            if id2 not in duplicates_to_remove:
                                duplicates_to_remove.add(id2)

    output_dir = Path(output_directory)
    for id in range(len(input_files)):
        if id in duplicates_to_remove: continue
        output_file = output_dir / Path(input_files[id]).name
        with open(input_files[id], "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out :
            for line in f_in:  # 逐行读取，防止大文件爆内存
                f_out.write(line)  # 原样写入，保留原格式

    # raise NotImplementedError

