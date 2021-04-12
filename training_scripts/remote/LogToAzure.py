import tensorflow.keras as K

class LogToAzure(K.callbacks.Callback):

    def __init__(self, run):
        super(LogToAzure, self).__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        for k, v in logs.items():
            # self.run.log_row(name=+k,
            #                  epoch=epoch,
            #                  value=v)
            self.run.log(k, v)
