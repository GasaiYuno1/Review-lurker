"""Оценки аномальности внутри темы и метрики ранжирования."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lurker.anomaly import (assign_topics, centroid_score, coherence_score, knn_score, lof_score, offtopic_score, precision_at_k,  # noqa: E402
                            ranking_metrics, recall_at_k, relevance_score, robust_z, turn_score)


def unit(M):
    return M / np.linalg.norm(M, axis=1, keepdims=True)


class AnomalyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        centers = unit(rng.normal(size=(3, 32)))
        self.topics = np.repeat([0, 1, 2], 60)
        E = centers[self.topics] + 0.15 * rng.normal(size=(180, 32))
        self.anom = np.zeros(180, dtype=bool)
        self.anom[[5, 70, 150]] = True
        E[self.anom] = unit(rng.normal(size=(3, 32)))      # случайное направление — чужая тема
        self.E = unit(E)

    def test_scores_rank_anomalies_first(self):
        for fn in (centroid_score, knn_score, lof_score):
            s = robust_z(fn(self.E, self.topics), self.topics)
            self.assertEqual(set(np.argsort(-s)[:3]), {5, 70, 150}, fn.__name__)

    def test_assign_topics_fills_outliers(self):
        lab = self.topics.copy()
        lab[[1, 100]] = -1
        out = assign_topics(self.E, lab)
        self.assertEqual(out[1], 0)
        self.assertEqual(out[100], 1)

    def test_turn_score_catches_single_bad_turn(self):
        rng = np.random.default_rng(1)
        base = unit(rng.normal(size=(4, 16)))              # 4 типовые реплики операторов
        T, doc = [], []
        for d in range(40):
            for j in range(4):
                T.append(base[j] + 0.05 * rng.normal(size=16))
                doc.append(d)
        T = np.array(T)
        T[4 * 7 + 2] = rng.normal(size=16)                 # в звонке 7 одна странная реплика
        T = unit(T)
        s = turn_score(T, np.array(doc), np.zeros(40, dtype=int), k=5)
        self.assertEqual(int(np.argmax(s)), 7)

    def test_offtopic_catches_spliced_segment(self):
        rng = np.random.default_rng(1)
        centers = unit(rng.normal(size=(2, 32)))
        doc_topics = np.repeat([0, 1], 30)
        turns, owner = [], []
        for d, t in enumerate(doc_topics):
            for j in range(8):                          # звонок 5 (тема 0): середина взята из разговора темы 1
                src = 1 if (d == 5 and 2 <= j < 5) else t
                turns.append(centers[src] + 0.3 * rng.normal(size=32))
                owner.append(d)
        s = offtopic_score(unit(np.array(turns)), np.array(owner), doc_topics)
        self.assertEqual(int(np.argmax(s)), 5)

    def test_coherence_and_relevance(self):
        rng = np.random.default_rng(2)
        a, b = unit(rng.normal(size=(2, 16)))
        T, doc, spk = [], [], []
        for d in range(10):
            for j in range(6):
                v = (b if (d == 3 and j in (2, 3)) else a) + 0.05 * rng.normal(size=16)   # звонок 3: середина о другом
                T.append(v); doc.append(d); spk.append("cl" if j % 2 == 0 else "op")
        T = unit(np.array(T))
        self.assertEqual(int(np.argmax(coherence_score(T, np.array(doc), 10))), 3)
        T2 = T.copy()
        T2[5 * 6 + 3] = unit(rng.normal(size=(1, 16)))[0]                                 # звонок 5: ответ невпопад
        self.assertEqual(int(np.argmax(relevance_score(T2, np.array(doc), spk, 10))), 5)

    def test_metrics(self):
        s = np.array([0.9, 0.8, 0.1, 0.7, 0.2])
        pos = np.array([1, 0, 0, 1, 0], dtype=bool)
        self.assertEqual(precision_at_k(s, pos, 2), 0.5)
        self.assertEqual(recall_at_k(s, pos, 3), 1.0)
        m = ranking_metrics(s, pos, ks=(2,), corrupted=np.array([0, 1, 0, 0, 0], dtype=bool))
        self.assertEqual(m["noise@2"], 0.5)
        self.assertAlmostEqual(m["auc"], 5 / 6)   # отрицательный 0.8 стоит выше положительного 0.7
        z = robust_z(np.array([1.0, 1.1, 0.9, 1.0, 5.0]), np.zeros(5))
        self.assertGreater(z[-1], 10)


if __name__ == "__main__":
    unittest.main()
