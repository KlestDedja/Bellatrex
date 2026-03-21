from dataclasses import dataclass
from typing import Callable


@dataclass(slots=True)
class TreeCacheEntry:
    """Cached tree render metadata for one main NiceGUI session."""

    tree_name: str
    image_name: str
    image_path: str
    image_source: str
    title: str


class TreeRenderCache:
    """Session-local cache for rendered tree images."""

    def __init__(self) -> None:
        self._entries: dict[str, TreeCacheEntry] = {}

    def get(self, tree_name: str) -> TreeCacheEntry | None:
        return self._entries.get(str(tree_name))

    def store(self, tree_name: str, entry: TreeCacheEntry) -> TreeCacheEntry:
        self._entries[str(tree_name)] = entry
        return entry

    def get_or_create(
        self,
        tree_name: str,
        builder: Callable[[str], TreeCacheEntry],
    ) -> tuple[TreeCacheEntry, bool]:
        normalized_tree_name = str(tree_name)
        entry = self.get(normalized_tree_name)
        if entry is not None:
            return entry, False

        entry = builder(normalized_tree_name)
        self.store(normalized_tree_name, entry)
        return entry, True

    def image_paths(self) -> list[str]:
        return [entry.image_path for entry in self._entries.values()]
