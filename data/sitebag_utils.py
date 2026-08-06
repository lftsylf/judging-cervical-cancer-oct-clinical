"""
Site-bag 协议工具：病理点级 CIN2+ 解析、钟点→TIFF、兄妹阳点共享。
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

POS_PAT = re.compile(
    r"(?:"
    r"HSIL|"
    r"CIN[-\s]*I{2,3}(?![IVX])|"
    r"CIN\s*[ⅡⅢ]|"
    r"CIN[-\s]*[2-3](?!\d)|"
    r"CINⅡ|CINⅢ|"
    r"高级别|"
    r"鳞状细胞癌|鳞癌|浸润癌|原位癌|腺癌|"
    r"恶性肿瘤|癌变"
    r")",
    re.I,
)


def canon_oct_id(s: str) -> str:
    s = str(s).strip().split("/")[0]
    m = re.match(r"(M\d+_\d+_P)(\d+)$", s)
    if not m:
        return s
    return f"{m.group(1)}{int(m.group(2))}"


def oct_id_numeric(s: str) -> Optional[int]:
    m = re.search(r"_P(\d+)$", str(s).strip().split("/")[0])
    return int(m.group(1)) if m else None


def expand_sites(token: str) -> List[int]:
    token = token.strip()
    out: List[int] = []
    mr = re.match(r"^(\d+)\s*[-–—~至到]\s*(\d+)$", token)
    if mr:
        a, b = int(mr.group(1)), int(mr.group(2))
        a = 12 if a == 0 else a
        b = 12 if b == 0 else b
        if 1 <= a <= 12 and 1 <= b <= 12:
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(list(range(a, 13)) + list(range(1, b + 1)))
        return out
    mn = re.match(r"^(\d+)$", token)
    if mn:
        v = int(mn.group(1))
        if v == 0:
            v = 12
        if 1 <= v <= 12:
            out.append(v)
    return out


def sites_from_numchunk(chunk: str) -> List[int]:
    parts = re.split(r"[、,，/\s]+", chunk.strip())
    sites: List[int] = []
    for p in parts:
        if p:
            sites.extend(expand_sites(p))
    return sorted(set(s for s in sites if 1 <= s <= 12))


def sites_from_header(header: str) -> List[int]:
    h = header.replace("点", " ").replace("°", " ").replace("度", " ")
    sites: List[int] = []
    for mobj in re.finditer(r"宫颈\s*((?:\d+\s*[、,，/\-–—~至到]*\s*)+)", h):
        sites.extend(sites_from_numchunk(mobj.group(1)))
    if sites:
        return sorted(set(sites))
    if re.search(r"宫颈|活检|穹", h):
        h2 = re.sub(r"组织\s*\d+\s*粒", "", h)
        h2 = re.sub(r"\d+\s*\*\s*\d+", "", h2)
        for num in re.findall(r"\d+", h2):
            sites.extend(expand_sites(num))
    return sorted(set(s for s in sites if 1 <= s <= 12))


def split_pathology_sections(text: str) -> List[Tuple[str, str]]:
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(
        re.finditer(
            r"(?:^|\n)\s*(?:"
            r"\d+\s*[\.．、]|"
            r"[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]|"
            r"[（(]\s*\d+\s*[）)]"
            r")\s*",
            text,
        )
    )
    if len(matches) >= 2:
        parts = []
        for i, mo in enumerate(matches):
            start = mo.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk = text[start:end].strip()
            first_nl = chunk.find("\n")
            header = chunk[:first_nl] if first_nl != -1 else chunk[:100]
            parts.append((header, chunk))
        return parts
    blocks = list(re.finditer(r"[（(][^）)]*(?:宫颈|阴道|穹|颈管)[^）)]*[）)]", text))
    if blocks:
        parts = []
        for i, mo in enumerate(blocks):
            start = mo.start()
            end = blocks[i + 1].start() if i + 1 < len(blocks) else len(text)
            chunk = text[start:end].strip()
            parts.append((mo.group(0), chunk))
        return parts
    return [("", text)]


def parse_xiangya_positive_sites(text: str) -> Tuple[List[int], str]:
    """湘雅长病理正文 → (阳点列表, 备注)。"""
    if not isinstance(text, str) or not text.strip():
        return [], "empty"
    sections = split_pathology_sections(text)
    pos = set()
    kw_no_site = False
    for header, body in sections:
        if not POS_PAT.search(body):
            continue
        sites = sites_from_header(header)
        if not sites:
            sites = sites_from_header(body[:80])
        if sites:
            pos.update(sites)
        else:
            kw_no_site = True
    if pos:
        return sorted(pos), "ok"
    if kw_no_site or POS_PAT.search(text):
        return [], "kw_no_site"
    return [], "no_pos"


def parse_liaoning_positive_sites(text: str) -> Tuple[List[int], str]:
    """辽宁紧凑病理 / IHC → (阳点列表, 备注)。"""
    if not isinstance(text, str) or not str(text).strip():
        return [], "empty"
    t = re.sub(r"^病理", "", text.replace("\r", "\n"))
    pos = set()
    anchors = list(re.finditer(r"((?:\d+\s*[、,，\-–—~至到]*\s*)+)\s*点", t))
    if anchors:
        for i, mo in enumerate(anchors):
            body = t[mo.end() : anchors[i + 1].start() if i + 1 < len(anchors) else len(t)]
            sites = sites_from_numchunk(mo.group(1))
            if POS_PAT.search(body) and sites:
                pos.update(sites)
        if pos:
            return sorted(pos), "ok"
    anchors = list(re.finditer(r"宫颈\s*((?:\d+\s*[、,，\-–—~至到]*\s*)+)\s*点", t))
    if anchors:
        for i, mo in enumerate(anchors):
            body = t[mo.end() : anchors[i + 1].start() if i + 1 < len(anchors) else len(t)]
            sites = sites_from_numchunk(mo.group(1))
            if POS_PAT.search(body) and sites:
                pos.update(sites)
        if pos:
            return sorted(pos), "ok"
    for mo in re.finditer(
        r"((?:\d+\s*[、,，\-–—~至到]*\s*)+)(?:点)?\s*([^，,；;。\n]{0,40})",
        t,
    ):
        if POS_PAT.search(mo.group(2)):
            pos.update(sites_from_numchunk(mo.group(1)))
    if pos:
        return sorted(pos), "ok"
    if POS_PAT.search(t):
        return [], "kw_no_site"
    return [], "no_pos"


def parse_huaxi_positive_sites(path_text: str, biopsy_text: str = "") -> Tuple[List[int], str]:
    """华西：优先病理正文；必要时从活检点旁路（仅当病理整段阳且活检点可解析时不自动全标）。"""
    sites, note = parse_xiangya_positive_sites(path_text) if isinstance(path_text, str) else ([], "empty")
    if sites:
        return sites, note
    # 华西有时写成「宫颈4-8点，…HSIL」类 compact
    if isinstance(path_text, str) and path_text.strip():
        sites2, note2 = parse_liaoning_positive_sites(path_text)
        if sites2:
            return sites2, note2
        if note2 == "kw_no_site" or note == "kw_no_site":
            return [], "kw_no_site"
    return [], note if note != "empty" else "no_pos"


def share_sibling_pos_sites(
    oct_ids: Sequence[str],
    labels: Sequence[int],
    pos_sites_list: Sequence[List[int]],
) -> List[List[int]]:
    """
    阳患者若自身无阳点，且存在 ID 数值 ±1 的兄妹、对方有阳点，则拷贝。
    """
    out = [list(s) for s in pos_sites_list]
    by_num: Dict[int, List[int]] = {}
    nums = []
    for i, oid in enumerate(oct_ids):
        n = oct_id_numeric(oid)
        nums.append(n)
        if n is not None:
            by_num.setdefault(n, []).append(i)

    for i, oid in enumerate(oct_ids):
        if int(labels[i]) != 1:
            continue
        if out[i]:
            continue
        n = nums[i]
        if n is None:
            continue
        for dn in (-1, 1):
            for j in by_num.get(n + dn, []):
                if int(labels[j]) == 1 and out[j]:
                    out[i] = list(out[j])
                    break
            if out[i]:
                break
    return out


def list_clock_to_tiff(img_folder: str) -> Dict[int, str]:
    """扫描患者目录，返回 {钟点 1..12: tiff 绝对或相对文件名}。"""
    mapping: Dict[int, str] = {}
    if not os.path.isdir(img_folder):
        return mapping
    for f in os.listdir(img_folder):
        fl = f.lower()
        if not (fl.endswith(".tiff") or fl.endswith(".tif")):
            continue
        if "duplicated" in fl:
            continue
        m = re.search(r"_C(\d+)_S\d+", f, re.I)
        if not m:
            continue
        clock = int(m.group(1))
        if 1 <= clock <= 12:
            # 同钟点多文件时取字典序更小者
            prev = mapping.get(clock)
            if prev is None or f < prev:
                mapping[clock] = f
    return mapping


def choose_train_sites(
    available: Sequence[int],
    label: int,
    pos_sites: Sequence[int],
    n_bag: int,
    rng: np.random.RandomState,
) -> List[int]:
    """训练组袋：阴随机；阳优先阳点。"""
    avail = [int(x) for x in available]
    if not avail:
        return []
    n_bag = min(int(n_bag), len(avail))
    pos = [p for p in pos_sites if p in avail]
    neg = [a for a in avail if a not in pos]

    if int(label) != 1 or not pos:
        # 阴或阳但位点未知
        if len(avail) <= n_bag:
            return sorted(avail)
        idx = rng.choice(len(avail), size=n_bag, replace=False)
        return sorted(int(avail[i]) for i in idx)

    if len(pos) >= n_bag:
        idx = rng.choice(len(pos), size=n_bag, replace=False)
        return sorted(int(pos[i]) for i in idx)

    # 阳点不足：全收阳点，再补阴点
    chosen = list(pos)
    need = n_bag - len(chosen)
    if need > 0 and neg:
        if len(neg) <= need:
            chosen.extend(neg)
        else:
            idx = rng.choice(len(neg), size=need, replace=False)
            chosen.extend(int(neg[i]) for i in idx)
    elif need > 0:
        rest = [a for a in avail if a not in chosen]
        chosen.extend(rest[:need])
    return sorted(chosen[:n_bag])


def iter_eval_windows(available: Sequence[int], n_bag: int) -> List[List[int]]:
    """验证/测试：按钟点排序不重叠切窗；n=2 时 12 点 → 6 窗。"""
    clocks = sorted(int(x) for x in available)
    if not clocks:
        return []
    n_bag = max(1, int(n_bag))
    windows = []
    for i in range(0, len(clocks), n_bag):
        w = clocks[i : i + n_bag]
        if w:
            windows.append(w)
    return windows


def encode_hpv(text) -> int:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return 0
    s = str(text).strip()
    if not s or s.upper() == "NAN":
        return 0
    if re.search(r"阴\s*性|（\s*-\s*）|\(-\)|DNA\s*检测\s*-|^/-$", s):
        if not re.search(r"\+|阳", s):
            return 0
    if re.search(r"\+|阳", s):
        return 1
    return 0


def encode_tct(text) -> int:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return 0
    s = str(text).strip().upper()
    if not s or s == "NAN":
        return 0
    if "HSIL" in s or "ASC-H" in s or "ASC_H" in s:
        return 4 if "HSIL" in s else 3
    if "LSIL" in s:
        return 2
    if "ASC-US" in s or "ASCUS" in s or "ASC_US" in s:
        return 1
    if "NILM" in s or "阴性" in str(text):
        return 0
    return 0


def encode_age(val, default: float = 45.0) -> float:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return float(default)
        s = str(val).strip().upper()
        if not s or s == "NAN":
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def sites_to_str(sites: Sequence[int]) -> str:
    return ",".join(str(int(s)) for s in sites)


def sites_from_str(s) -> List[int]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    text = str(s).strip()
    if not text or text.lower() == "nan":
        return []
    out = []
    for part in re.split(r"[,\s]+", text):
        if part.isdigit():
            v = int(part)
            if 1 <= v <= 12:
                out.append(v)
    return sorted(set(out))
