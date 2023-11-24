class BatchIndex:
    def __init__(self, size, batch_size, shuffle=False, drop_last=False):
        if drop_last:
            self.index_list = [(x, x + batch_size,) for x in range(size) if
                               not x % batch_size and x + batch_size <= size]
        else:
            li = [(x, x + batch_size,) for x in range(size) if not x % batch_size]
            x, y = li[-1]
            li[-1] = (x, size)
            self.index_list = li
            self.shuffle = shuffle
            self.drop_last = drop_last

        if shuffle:
            import random as r
            r.shuffle(self.index_list)

    def __next__(self):
        self.pos += 1
        if self.pos >= len(self.index_list):
            raise StopIteration

        return self.index_list[self.pos]

    def __iter__(self):
        self.pos = -1
        return self

    def __len__(self):
        return len(self.index_list)
