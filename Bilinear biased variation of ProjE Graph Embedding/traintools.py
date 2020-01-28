

class NegativeGenerator(object): #@# not mine
    def __init__(self, n_ent, n_negative, train_graph=None):
        self.n_ent = n_ent
        self.n_negative = n_negative
        if train_graph:
            raise NotImplementedError
        self.graph = train_graph  # for preventing from including positive triplets as negative ones

    def generate(self, pos_triplets):
        """
        :return: neg_triplets, whose size is (length of positives \times n_sample , 3)
        """
        raise NotImplementedError


class UniformNegativeGenerator(NegativeGenerator): #@# not mine
    def __init__(self, n_ent, n_negative, train_graph=None):
        super(UniformNegativeGenerator, self).__init__(n_ent, n_negative, train_graph)

    def generate(self, pos_triplets):
        _batchsize = len(pos_triplets)
        sample_size = _batchsize * self.n_negative
        neg_ents = np.random.randint(0, self.n_ent, size=sample_size)
        neg_triplets = np.tile(pos_triplets, (self.n_negative, 1))
        head_or_tail = 2 * np.random.randint(0, 2, sample_size)
        neg_triplets[np.arange(sample_size), head_or_tail] = neg_ents
        return neg_triplets
