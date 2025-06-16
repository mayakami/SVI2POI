class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  # Â·¾¶Ñ¹Ëõ
        return self.parent[item]

    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0

    def get_groups(self):
        groups = {}
        for item in self.parent:
            root = self.find(item)
            if root not in groups:
                groups[root] = []
            groups[root].append(item)
        return groups

    def find_group(self, item):
        if item not in self.parent:
            return None  # Item not found in any group
        root = self.find(item)
        return [member for member, parent in self.parent.items() if self.find(parent) == root]