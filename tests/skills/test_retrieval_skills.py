from agentic_rag.math_retriever import MathKnowledgeRetriever


class FakeCollection:
    ids = ["d1", "d2", "d3"]
    documents = [
        "一次函数依赖平面直角坐标系，斜率表示变化率。",
        "三角形内角和为180度。",
        "有理数包含正数、负数和零。",
    ]
    metadatas = [{"chapter": "函数"}, {"chapter": "几何"}, {"chapter": "代数"}]

    def get(self, **_kwargs):
        return {"ids": self.ids, "documents": self.documents, "metadatas": self.metadatas}

    def query(self, **_kwargs):
        return {"ids": [["d1", "d2"]], "documents": [self.documents], "metadatas": [self.metadatas], "distances": [[0.1, 0.8]]}


def retriever():
    value = MathKnowledgeRetriever("unused")
    value._collection = FakeCollection()
    return value


def test_dense_channel_only_labels_dense_results():
    documents, _ = retriever().retrieve_dense_channel(["一次函数斜率"], "综合", 2)
    assert [item.metadata["retrieval_channel"] for item in documents] == ["dense", "dense"]


def test_bm25_channel_is_lexical_and_does_not_mix_dense_labels():
    documents, _ = retriever().retrieve_bm25_channel(["斜率"], "综合", 2)
    assert documents
    assert all(item.metadata["retrieval_channel"] == "bm25" for item in documents)


def test_graph_channel_uses_prerequisite_expansion():
    documents, trace = retriever().retrieve_graph_channel(["一次函数"], "综合", 2)
    assert documents[0].metadata["retrieval_channel"] == "graph"
    assert "平面直角坐标系" in trace[0]["expanded_points"]
