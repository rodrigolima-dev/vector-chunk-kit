import pytest

from vector_chunk_kit.prepare import DEFAULT_TEMPLATE, prepare_documents


def document(**changes: object) -> dict[str, object]:
    item: dict[str, object] = {
        "id": "guide-1",
        "namespace": "demo",
        "title": "Plant care",
        "text": "Water seedlings gently. Keep the soil damp. Give them morning light.",
        "keywords": ["seedlings", "soil"],
        "metadata": {"section": "basics"},
    }
    item.update(changes)
    return item


def test_prepare_is_deterministic_and_includes_custom_context() -> None:
    template = "Topic: {title}\nTags: {keywords}\n{body}"
    first = prepare_documents([document()], "demo", 120, 20, template)
    second = prepare_documents([document()], "demo", 120, 20, template)

    assert first == second
    assert len(first) == 1
    assert first[0].chunk_id
    assert first[0].content_hash
    assert first[0].text.startswith("Topic: Plant care\nTags: seedlings, soil")
    assert first[0].metadata == {"section": "basics"}


def test_metadata_change_invalidates_a_prepared_record() -> None:
    before = prepare_documents([document()], "demo", 120, 0, DEFAULT_TEMPLATE)
    after = prepare_documents(
        [document(metadata={"section": "advanced"})], "demo", 120, 0, DEFAULT_TEMPLATE
    )
    assert before[0].chunk_id == after[0].chunk_id
    assert before[0].content_hash != after[0].content_hash


def test_prepare_splits_text_with_overlap_without_losing_words() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    chunks = prepare_documents([document(text=text)], "demo", 24, 10, "{body}")

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 24 for chunk in chunks)
    assert chunks[0].text.split()[-1] in chunks[1].text.split()
    assert chunks[-1].text.endswith("lambda")


@pytest.mark.parametrize(
    "item,reason",
    [
        (document(namespace="another"), "namespace"),
        (document(text=""), "text"),
        (document(keywords="hidden"), "keywords"),
        (document(metadata=["wrong"]), "metadata"),
        (document(id="../unsafe"), "id"),
    ],
)
def test_prepare_rejects_invalid_or_cross_namespace_input(
    item: dict[str, object], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        prepare_documents([item], "demo", 80, 10, DEFAULT_TEMPLATE)


def test_prepare_rejects_duplicate_source_id() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        prepare_documents([document(), document()], "demo", 100, 0, DEFAULT_TEMPLATE)


def test_prepare_rejects_template_with_unknown_field() -> None:
    with pytest.raises(ValueError, match="template"):
        prepare_documents([document()], "demo", 100, 0, "{private_field}")
