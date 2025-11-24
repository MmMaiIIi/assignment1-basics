use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

/// py_words: list of ( [chunk_bytes...], count )
/// vocab_size: 总 vocab 大小
/// num_special_tokens: 特殊 token 数量
///
/// 返回：按顺序的 merges 列表，每个是 (b1, b2) —— 在 Python 侧是 (bytes, bytes)
#[pyfunction]
fn run_bpe_merge(
    py: Python<'_>,
    py_words: Vec<(Vec<Vec<u8>>, usize)>,
    vocab_size: usize,
    num_special_tokens: usize,
) -> PyResult<Vec<(Py<PyBytes>, Py<PyBytes>)>> {
    // words_encoding: word(bytes list) -> count
    let mut words_encoding: HashMap<Vec<Vec<u8>>, usize> = HashMap::new();
    for (word, count) in py_words {
        *words_encoding.entry(word).or_insert(0) += count;
    }

    // 和 Python 一样：vocab_size - 256 - len(special_tokens)
    let merges_iter_number = vocab_size.saturating_sub(256 + num_special_tokens);

    // pair_freq: (b1, b2) -> freq
    let mut pair_freq: HashMap<(Vec<u8>, Vec<u8>), usize> = HashMap::new();
    // pair_to_words: (b1, b2) -> set(words containing this pair)
    let mut pair_to_words: HashMap<(Vec<u8>, Vec<u8>), HashSet<Vec<Vec<u8>>>> = HashMap::new();

    // 初始化 pair_freq 和 pair_to_words
    for (word, &count) in words_encoding.iter() {
        if word.len() < 2 {
            continue;
        }
        for i in 0..(word.len() - 1) {
            let pair = (word[i].clone(), word[i + 1].clone());
            *pair_freq.entry(pair.clone()).or_insert(0) += count;
            pair_to_words.entry(pair).or_default().insert(word.clone());
        }
    }

    // 内部先存 Vec<u8>，方便操作
    let mut merges_bytes: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

    for _ in 0..merges_iter_number {
        if pair_freq.is_empty() {
            break;
        }

        // 找到 freq 最大的 pair，freq 相同按 pair 字典序打破平局
        let (best_pair, _) = pair_freq
            .iter()
            .max_by(|(p1, f1), (p2, f2)| match f1.cmp(f2) {
                Ordering::Less => Ordering::Less,
                Ordering::Greater => Ordering::Greater,
                Ordering::Equal => {
                    let (a0, a1) = (&p1.0, &p1.1);
                    let (b0, b1) = (&p2.0, &p2.1);
                    let first = a0.cmp(b0);
                    if first != Ordering::Equal {
                        first
                    } else {
                        a1.cmp(b1)
                    }
                }
            })
            .unwrap();

        let max_key = best_pair.clone();
        merges_bytes.push((max_key.0.clone(), max_key.1.clone()));

        // 找到所有包含 max_key 的 word
        let Some(affecting_words) = pair_to_words.remove(&max_key) else {
            continue;
        };

        for word in affecting_words {
            // 取出旧的 count
            let Some(count) = words_encoding.remove(&word) else {
                continue;
            };

            // 1) 从 pair_freq / pair_to_words 里减去旧 word 的贡献
            if word.len() >= 2 {
                for i in 0..(word.len() - 1) {
                    let pair = (word[i].clone(), word[i + 1].clone());

                    // 更新 pair_freq
                    let remove_pair = if let Some(freq) = pair_freq.get_mut(&pair) {
                        *freq = freq.saturating_sub(count);
                        *freq == 0
                    } else {
                        false
                    };
                    if remove_pair {
                        pair_freq.remove(&pair);
                    }

                    // 更新 pair_to_words
                    let remove_entry = if let Some(words_set) = pair_to_words.get_mut(&pair) {
                        words_set.remove(&word);
                        words_set.is_empty()
                    } else {
                        false
                    };
                    if remove_entry {
                        pair_to_words.remove(&pair);
                    }
                }
            }

            // 2) 构造 new_key：把所有 max_key 合并
            let mut new_key: Vec<Vec<u8>> = Vec::new();
            let mut i = 0usize;
            while i + 1 < word.len() {
                let pair = (word[i].clone(), word[i + 1].clone());
                if pair == max_key {
                    let mut merged = word[i].clone();
                    merged.extend_from_slice(&word[i + 1]);
                    new_key.push(merged);
                    i += 2;
                } else {
                    new_key.push(word[i].clone());
                    i += 1;
                }
            }
            if i < word.len() {
                new_key.push(word[i].clone());
            }

            // 3) 用 new_key 把新 pair 的贡献加回去
            if new_key.len() >= 2 {
                for j in 0..(new_key.len() - 1) {
                    let pair = (new_key[j].clone(), new_key[j + 1].clone());
                    *pair_freq.entry(pair.clone()).or_insert(0) += count;
                    pair_to_words
                        .entry(pair)
                        .or_default()
                        .insert(new_key.clone());
                }
            }

            // 4) 更新 words_encoding
            *words_encoding.entry(new_key).or_insert(0) += count;
        }
    }

    // === 关键：把 Vec<u8> 转成 Python bytes ===
    let result: Vec<(Py<PyBytes>, Py<PyBytes>)> = merges_bytes
        .into_iter()
        .map(|(b1, b2)| {
            (
                PyBytes::new_bound(py, &b1).into(),
                PyBytes::new_bound(py, &b2).into(),
            )
        })
        .collect();

    Ok(result)
}

/// PyO3 0.22+ 的模块初始化函数，名字必须是 fast_bpe
#[pymodule]
fn fast_bpe(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_bpe_merge, m)?)?;
    Ok(())
}
