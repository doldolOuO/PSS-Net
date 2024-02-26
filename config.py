class params:
    def __init__(self): 
        
        #General Parameters
        self.cat = "chair"
        # plane cabinet car chair lamp couch table watercraft
        # bench monitor speaker firearm cellphone
        
        #Decoder parameters
        self.num_branch = 8 # max=15
        
        #Training parameters
        self.batch_size = 32
        self.nThreads = 8
        self.lr = 0.0001
        self.dataroot = "/home/user/FCH/data/ShapeNetViPC_2048"
        self.n_epochs =100
        self.eval_epoch = 1
        self.resume = False
        self.lr_fix_epoch = 50
        self.steps = 50000