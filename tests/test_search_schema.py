"""Tests for the Azure AI Search index schema."""

import pytest
from azure.search.documents.indexes.models import SearchFieldDataType

from src.app.azure_search_store import (
    EMBEDDING_DIMENSIONS,
    HNSW_ALGORITHM_NAME,
    VECTOR_FIELD_NAME,
    VECTOR_PROFILE_NAME,
    build_search_index,
)


def test_schema_contains_required_document_fields() -> None:
    index = build_search_index("assistant-chunks")
    fields = {field.name: field for field in index.fields}

    assert set(fields) == {
        "id",
        "content",
        "source",
        "page",
        "chunk_id",
        VECTOR_FIELD_NAME,
    }
    assert fields["id"].key is True
    assert fields["content"].searchable is True
    assert fields["source"].filterable is True
    assert fields["page"].filterable is True
    assert fields["chunk_id"].filterable is True


def test_vector_field_uses_expected_dimensions_and_profile() -> None:
    index = build_search_index("assistant-chunks")
    fields = {field.name: field for field in index.fields}
    vector_field = fields[VECTOR_FIELD_NAME]

    assert vector_field.type == SearchFieldDataType.Collection(
        SearchFieldDataType.Single
    )
    assert vector_field.vector_search_dimensions == EMBEDDING_DIMENSIONS
    assert vector_field.vector_search_profile_name == VECTOR_PROFILE_NAME
    assert vector_field.searchable is True
    assert vector_field.retrievable is False


def test_schema_uses_hnsw_with_cosine_similarity() -> None:
    index = build_search_index("assistant-chunks")
    vector_search = index.vector_search

    assert vector_search is not None
    assert len(vector_search.algorithms) == 1
    assert vector_search.algorithms[0].name == HNSW_ALGORITHM_NAME
    assert vector_search.algorithms[0].parameters.metric == "cosine"
    assert vector_search.profiles[0].name == VECTOR_PROFILE_NAME
    assert (
        vector_search.profiles[0].algorithm_configuration_name
        == HNSW_ALGORITHM_NAME
    )


@pytest.mark.parametrize("index_name", ["", "   "])
def test_schema_rejects_blank_index_names(index_name: str) -> None:
    with pytest.raises(ValueError, match="index_name must not be empty"):
        build_search_index(index_name)
