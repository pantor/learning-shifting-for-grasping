import time

import tensorflow as tf
from tqdm import tqdm_notebook as tqdm


class Model:
    def __init__(self, test_metrices=[], inputs={}, outputs={}, model_input_path=None, model_output_path=None, import_graph=False, import_weights=False):
        self.test_metrices = test_metrices
        self.inputs = inputs
        self.outputs = outputs
        self.model_input_path = model_input_path
        self.model_output_path = model_output_path
        
        self.print_only_on_best_model = False
        
        self.graph = tf.Graph() if import_graph else tf.get_default_graph()
        with self.graph.as_default(): 
            self.saver = tf.train.import_meta_graph(self.model_input_path + '.meta') if import_graph else tf.train.Saver()
        self.sess = tf.Session(graph=self.graph)
        if import_weights:
            self.load()
 
    def __del__(self):
        self.sess.close()

    def printEvaluation(self, i, train_acc, test_data):
        if (i == -1):
            print('Epoch\tTime[s]\tTrain Accuracy\tTest Loss\tAccuracy\tPrecision\tRecall\t\tF1')
    
        dt = time.time() - self.start
        print('{:5d}\t{:0.2f}\t{:0.4f}\t\t{:0.4f}\t\t{:0.4f}\t\t{:0.4f}\t\t{:0.4f}\t\t{:0.4f}'.format(
            i, dt, train_acc, test_data[0], test_data[1], test_data[2], test_data[3], test_data[4]
        ))
        
    def onTrainStart(self):
        self.train_handle = self.sess.run(self.train_iterator.string_handle())
        self.test_handle = self.sess.run(self.test_iterator.string_handle())
        
        self.sess.run(self.test_iterator.initializer)
        self.test_min_data = self.sess.run(self.test_metrices, feed_dict={'handle:0': self.test_handle})
        self.epoch_best = 0
        
        self.start = time.time()
        self.printEvaluation(-1, -1, self.test_min_data)
        
    def onTrainEnd(self):
        print('Training stopped.')

    def onEpochStart(self):
        self.sess.run(self.train_iterator.initializer)
    
    def onEpochEnd(self, epoch, train_data, early_stopping_patience, save):
        self.sess.run(self.test_iterator.initializer)
        
        test_data = self.sess.run(self.test_metrices, feed_dict={'handle:0': self.test_handle})
        
        if not self.print_only_on_best_model:
            self.printEvaluation(epoch, train_data[1], test_data)
        
        min_index = 0
        if test_data[min_index] < self.test_min_data[min_index]:
            self.test_min_data[min_index] = test_data[min_index]
            self.epoch_best = epoch
            
            if self.print_only_on_best_model:
                self.printEvaluation(epoch, train_data[1], test_data)
                    
            if save:
                self.save()
            else:
                print('Best model yet!')
                                
        # Early stopping
        if epoch > self.epoch_best + early_stopping_patience:
            return True
        
        # Reduce learning rate
        # if epoch > self.epoch_best + early_stopping_patience:
        
    def load(self):
        self.saver.restore(self.sess, self.model_input_path)
        
    def save(self):
        save_path = self.saver.save(self.sess, self.model_output_path)
        print('Model saved in file: {}'.format(save_path))
        
    def export(self):
        tf.saved_model.simple_save(self.sess, self.model_output_path + '-sm', inputs=self.inputs, outputs=self.outputs)
        
    def fit(self, train_iterator, train_batches_per_epoch, test_iterator, epochs=100, early_stopping_patience=0, load=False, save=False, export=False):
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
        
        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            
        if load:
            self.load()
            
        if export:
            self.export()
            return
        
        self.onTrainStart()
        for epoch in range(epochs):
            self.onEpochStart()
            
            for batch in range(train_batches_per_epoch):
#            for batch in tqdm(range(train_batches_per_epoch), unit='batches', leave=False, dynamic_ncols=True):
                train_data = self.sess.run(['train', self.test_metrices[1]], feed_dict={'handle:0': self.train_handle, 'training:0': True})
            
            if self.onEpochEnd(epoch, train_data, early_stopping_patience, save):
                return
        self.onTrainEnd()
        
    def predict(self, inputs):
        output_values = [str(x) for x in self.outputs.values()]
        input_values = [str(x) for x in self.inputs.values()]
        output_array = output_values[0] if len(output_values) == 1 else output_values
        input_array = {input_values[0]: inputs} if len(input_values) == 1 else dict(zip(input_values, inputs))
        return self.sess.run(output_array, feed_dict=input_array)