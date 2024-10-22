class params:
    def __init__(self): 
        #General Parameters
        self.cat = "all"
        # plane cabinet car chair lamp couch table watercraft all
        
        #Training parameters
        self.batch_size = 32
        self.nThreads = 8
        self.lr = 0.0001
        self.data_root = "/data/FCH/data/ShapeNetViPC_2048"
        self.n_epochs = 200
        self.eval_epoch = 1
        self.resume = False
        self.lr_fix_epoch = 50
        self.steps = 50000