import tensorflow as tf
from Camvid_Loader import DataLoader
from utils import cross_entropy, dice_coeff, count_parameters, compute_iou
import os
import numpy as np
from scipy import misc
import xlsxwriter
import matplotlib.pyplot as plt
class BaseModel(object):

    def __init__(self, sess, conf):
        self.sess = sess
        self.conf = conf
        self.is_training = True
        self.input_shape = [None, None, None, self.conf.channel]
        self.output_shape = [None, None, None]
        self.create_placeholders()
        x_flip = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), self.x)
        self.x_aug = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=32./255.), x_flip)
        self.x_input = tf.cond(self.isTraining,  # condition
                                  lambda: self.x_aug,  # if True (Training)
                                  lambda: self.x,  # if False (Test)
                                  name="input_augmentation")


    def create_placeholders(self):
        with tf.name_scope('Input'):
            self.x = tf.placeholder(tf.float32, self.input_shape, name='input')
            self.y = tf.placeholder(tf.int64, self.output_shape, name='annotation')
            self.isTraining = tf.placeholder_with_default(False, shape=(), name="isTraining")
            self.keep_prob = tf.placeholder(tf.float32)

    def loss_func(self):
        with tf.name_scope('Loss'):
            y_one_hot = tf.one_hot(self.y, depth=self.conf.num_cls, axis=3, name='y_one_hot')
            if self.conf.loss_type == 'cross-entropy':
                with tf.name_scope('cross_entropy'):
                    loss = cross_entropy(y_one_hot, self.logits, self.conf.num_cls)
            elif self.conf.loss_type == 'dice':
                with tf.name_scope('dice_coefficient'):
                    loss = dice_coeff(y_one_hot, self.logits)
            with tf.name_scope('L2_loss'):
                l2_loss = tf.reduce_sum(
                    self.conf.lmbda * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('reg_weights')]))
            with tf.name_scope('total'):
                self.total_loss = loss + l2_loss
                self.mean_loss, self.mean_loss_op = tf.metrics.mean(self.total_loss)

    def accuracy_func(self):
        with tf.name_scope('Accuracy'):
            self.y_pred = tf.argmax(self.logits, axis=3, name='decode_pred')
            correct_prediction = tf.equal(self.y, self.y_pred, name='correct_pred')
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_op')
            self.mean_accuracy, self.mean_accuracy_op = tf.metrics.mean(accuracy)

    def configure_network(self):
        self.loss_func()
        self.accuracy_func()
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(self.conf.init_lr,
                                                   global_step,
                                                   decay_steps=500,
                                                   decay_rate=0.97,
                                                   staircase=True)
        self.learning_rate = tf.maximum(learning_rate, self.conf.lr_min)
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.total_loss, global_step=global_step)
        self.sess.run(tf.global_variables_initializer())
        trainable_vars = tf.trainable_variables()
        self.saver = tf.train.Saver(var_list=trainable_vars, max_to_keep=1000)
        self.train_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/train/', self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(self.conf.logdir + self.conf.run_name + '/valid/')
        self.configure_summary()
        print('*' * 50)
        print('Total number of trainable parameters: {}'.
              format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        print('*' * 50)

    def configure_summary(self):
        summary_list = [tf.summary.scalar('learning_rate', self.learning_rate),
                        tf.summary.scalar('loss', self.mean_loss),
                        tf.summary.scalar('accuracy', self.mean_accuracy),
                        tf.summary.image('train/original_image',
                                         self.x[:, :, :],
                                         max_outputs=self.conf.batch_size),
                        tf.summary.image('train/prediction_mask',
                                         tf.cast(tf.expand_dims(self.y_pred[:, :, :], -1),
                                                 tf.float32),
                                         max_outputs=self.conf.batch_size),
                        tf.summary.image('train/original_mask',
                                         tf.cast(tf.expand_dims(self.y[:, :, :], -1), tf.float32),
                                         max_outputs=self.conf.batch_size)]
        self.merged_summary = tf.summary.merge(summary_list)

    def save_summary(self, summary, step):
        # print('----> Summarizing at step {}'.format(step))
        if self.is_training:
            self.train_writer.add_summary(summary, step)
        else:
            self.valid_writer.add_summary(summary, step)
        self.sess.run(tf.local_variables_initializer())

    def train(self):
        self.sess.run(tf.local_variables_initializer())
        self.best_validation_accuracy = 0
        self.best_mean_IOU = 0
        if self.conf.reload_step > 0:
            self.reload(self.conf.reload_step)
            print('----> Continue Training from step #{}'.format(self.conf.reload_step))
        else:
            print('----> Start Training')
        self.data_reader = DataLoader(self.conf)
        self.numValid = self.data_reader.count_num_samples(mode='valid')
        self.num_val_batch = int(self.numValid / self.conf.val_batch_size)
        for train_step in range(1, self.conf.max_step + 1):
            # print('Step: {}'.format(train_step))
            self.is_training = True
            if train_step % self.conf.SUMMARY_FREQ == 0:
                x_batch, y_batch = self.data_reader.next_batch(mode='train')
                feed_dict = {self.x: x_batch, self.y: y_batch, self.keep_prob: self.conf.drop_out_rate, self.isTraining:True}
                _, _, _, summary = self.sess.run([self.train_op,
                                                  self.mean_loss_op,
                                                  self.mean_accuracy_op,
                                                  self.merged_summary],
                                                 feed_dict=feed_dict)

                loss, acc = self.sess.run([self.mean_loss, self.mean_accuracy])
                print('step: {0:<6}, train_loss= {1:.4f}, train_acc={2:.01%}'.format(train_step + self.conf.reload_step, loss, acc))
                self.save_summary(summary, train_step + self.conf.reload_step)
            else:
                x_batch, y_batch = self.data_reader.next_batch(mode='train')
                feed_dict = {self.x: x_batch, self.y: y_batch, self.keep_prob: self.conf.drop_out_rate}
                self.sess.run([self.train_op, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            if train_step % self.conf.VAL_FREQ == 0:
                self.is_training = False
                self.evaluate(train_step)
    '''
            if train_step % self.conf.SAVE_FREQ == 0:
                self.save(train_step + self.conf.reload_step)

    def evaluate(self, train_step):
        self.sess.run(tf.local_variables_initializer())
        for step in range(self.num_val_batch):
            start = step * self.conf.val_batch_size
            end = (step + 1) * self.conf.val_batch_size
            x_val, y_val = self.data_reader.next_batch(start, end, mode='valid')
            feed_dict = {self.x: x_val, self.y: y_val, self.keep_prob: 1}
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)

        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss, valid_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        self.save_summary(summary_valid, train_step + self.conf.reload_step)
        print('-' * 30 + 'Validation' + '-' * 30)
        print('After {0} training step: val_loss= {1:.4f}, val_acc={2:.01%}'.format(train_step, valid_loss, valid_acc))
        print('-' * 70)

    def test(self, step_num):
        self.sess.run(tf.local_variables_initializer())
        self.reload(step_num)
        self.data_reader = DataLoader(self.conf)
        self.numTest = self.data_reader.count_num_samples(mode='test')
        self.num_test_batch = int(self.numTest / self.conf.val_batch_size)
        self.is_train = False
        self.sess.run(tf.local_variables_initializer())
        
        im_dir = os.path.join('./Results/test_prediction')
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
            
        for step in range(self.num_test_batch):
            start = step * self.conf.val_batch_size
            end = (step + 1) * self.conf.val_batch_size
            x_test, y_test = self.data_reader.next_batch(start, end, mode='test')
            feed_dict = {self.x: x_test, self.y: y_test, self.keep_prob: 1}
            mask_out, _,_ =self.sess.run([self.y_pred, self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            
            im = mask_out[0,:,:]
            cls1=np.zeros((self.conf.height,self.conf.width))
            cls2 = np.zeros((self.conf.height,self.conf.width))
            cls1[np.where(im==1)]=255
            cls2[np.where(im == 2)] = 255
            im_name = os.path.join(im_dir,str(1001+step)+'.jpg')
            cls1_name = os.path.join(im_dir, 'cls1_'+str(1001 + step) + '.jpg')
            cls2_name = os.path.join(im_dir, 'cls2_'+str(1001 + step) + '.jpg')
            misc.imsave(im_name, im)
            misc.imsave(cls1_name, cls1)
            misc.imsave(cls2_name, cls2)
        
        test_loss, test_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        print('-' * 18 + 'Test Completed' + '-' * 18)
        print('test_loss= {0:.4f}, test_acc={1:.01%}'.
              format(test_loss, test_acc))
        print('-' * 50)

    
    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
        checkpoint_path = os.path.join(self.conf.modeldir+self.conf.run_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir+self.conf.run_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        print('----> Model successfully restored')
    '''


    def evaluate(self, train_step):
        self.sess.run(tf.local_variables_initializer())
        self.num_valid = self.num_val_batch * self.conf.val_batch_size
        all_y = np.zeros((self.num_valid, self.conf.eval_height, self.conf.eval_width))
        all_y_pred = np.zeros((self.num_valid, self.conf.eval_height, self.conf.eval_width))
        for step in range(self.num_val_batch):
            start = step * self.conf.val_batch_size
            end = (step + 1) * self.conf.val_batch_size
            x_val, y_val = self.data_reader.next_batch(start, end, mode='valid')
            feed_dict = {self.x: x_val[:,:self.conf.eval_height,:self.conf.eval_width,:], self.y: y_val[:,:self.conf.eval_height,:self.conf.eval_width], self.keep_prob: 1, self.isTraining:False}
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            y, y_pred = self.sess.run([self.y, self.y_pred], feed_dict=feed_dict)
            #all_y = np.concatenate((all_y, y), axis=0)
            #all_y_pred = np.concatenate((all_y_pred, y_pred), axis=0)
            all_y[start:end, :, :] = y
            all_y_pred[start:end, :, :] = y_pred
        IOU = compute_iou(all_y_pred, all_y, num_cls=self.conf.num_cls)
        mean_IOU = np.mean(IOU[:-1])
        summary_valid = self.sess.run(self.merged_summary, feed_dict=feed_dict)
        valid_loss, valid_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        self.save_summary(summary_valid, train_step + self.conf.reload_step)
        if valid_acc > self.best_validation_accuracy:
            self.best_validation_accuracy = valid_acc
            improved_str = '(improved: accuracy)'
            if mean_IOU > self.best_mean_IOU:
                self.best_mean_IOU = mean_IOU
                improved_str = '(improved: accuracy & mean_IOU)'
            self.save(train_step + self.conf.reload_step)
        elif mean_IOU > self.best_mean_IOU:
            self.best_mean_IOU = mean_IOU
            improved_str = '(improved: mean_IOU)'
            self.save(train_step + self.conf.reload_step)
        else:
            improved_str = ''
        print('-' * 25 + 'Validation' + '-' * 25)
        print('After {0} training step: val_loss= {1:.4f}, val_acc={2:.01%}{3}'
              .format(train_step + self.conf.reload_step, valid_loss, valid_acc, improved_str))
        print('Building={1:.01%}, Tree={5:.01%}, Sky={0:.01%}, Car={8:.01%}, '
               'Sign={6:.01%}, Road={3:.01%}, Pedestrian={9:.01%}, Fence={7:.01%}, Pole={2:.01%}, '
               'Sidewalk={4:.01%}, Bicyclist={10:.01%}, void={11:.01%}, Average={12:.01%}'
               .format(IOU[0], IOU[1], IOU[2], IOU[3], IOU[4], IOU[5], IOU[6], IOU[7], IOU[8], IOU[9], IOU[10], IOU[11], mean_IOU))
        #print('background={0:.01%}, cells={1:.01%}, vessels={2:.01%}, mean_IOU={3:.01%}, '
         #     .format(IOU[0], IOU[1], IOU[2],mean_IOU))
        print('-' * 60)

    def test(self, step_num):

        self.sess.run(tf.local_variables_initializer())
        self.reload(step_num)
        self.data_reader = DataLoader(self.conf)
        self.numTest = self.data_reader.count_num_samples(mode='test')
        self.num_test_batch = int(self.numTest / self.conf.val_batch_size)
        self.num_test = self.num_test_batch * self.conf.val_batch_size
        self.is_train = False
        self.sess.run(tf.local_variables_initializer())
        all_y = np.zeros((self.num_test, self.conf.eval_height, self.conf.eval_width))
        all_y_pred = np.zeros((self.num_test, self.conf.eval_height, self.conf.eval_width))
        test_list = np.zeros((self.num_test_batch, 4))
        for step in range(self.num_test_batch):
            start = step * self.conf.val_batch_size
            end = (step + 1) * self.conf.val_batch_size
            x_test, y_test = self.data_reader.next_batch(start, end, mode='test')
            feed_dict = {self.x: x_test[:,:self.conf.eval_height,:self.conf.eval_width,:], self.y: y_test[:,:self.conf.eval_height,:self.conf.eval_width], self.keep_prob: 1, self.isTraining:False}
            self.sess.run([self.mean_loss_op, self.mean_accuracy_op], feed_dict=feed_dict)
            y, y_pred = self.sess.run([self.y, self.y_pred], feed_dict=feed_dict)
            test_loss, test_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
            IOU0 = compute_iou(y_pred, y, num_cls=self.conf.num_cls)
            mean_IOU0 = np.mean(IOU0[:-1])
            test_list[step,:] = [step, test_loss, test_acc, mean_IOU0]
            print('test#:{}, loss:{:0.2}, accuracy:{:.2%}, mean_IOU:{:.2%}'.format(step, test_loss, test_acc, mean_IOU0))
            all_y[start:end,:,:] = y
            all_y_pred[start:end, :, :] = y_pred

        IOU = compute_iou(all_y_pred, all_y, num_cls=self.conf.num_cls)
        mean_IOU = np.mean(IOU[:-1])
        test_loss, test_acc = self.sess.run([self.mean_loss, self.mean_accuracy])
        print('-' * 18 + 'Test Completed' + '-' * 18)
        print('test_loss= {0:.4f}, test_acc={1:.01%}'.format(test_loss, test_acc))
        print('Building={1:.01%}, Tree={5:.01%}, Sky={0:.01%}, Car={8:.01%}, '
               'Sign={6:.01%}, Road={3:.01%}, Pedestrian={9:.01%}, Fence={7:.01%}, Pole={2:.01%}, '
               'Sidewalk={4:.01%}, Bicyclist={10:.01%}, void={11:.01%}, Average={12:.01%}'
               .format(IOU[0], IOU[1], IOU[2], IOU[3], IOU[4], IOU[5], IOU[6], IOU[7], IOU[8], IOU[9], IOU[10], IOU[11],
                       mean_IOU))
        #print('background={0:.01%}, cells={1:.01%}, vessels={2:.01%}, mean_IOU={3:.01%}, '
        #      .format(IOU[0], IOU[1], IOU[2], mean_IOU))
        print('-' * 60)

        workbook = xlsxwriter.Workbook(os.path.join('valid_loss_acc.xlsx'))
        worksheet = workbook.add_worksheet()
        row = 0
        col = 0
        for step, test_loss, test_acc, mean_IOU0 in (test_list):
            worksheet.write(row, col, step)
            worksheet.write(row, col + 1, test_loss)
            worksheet.write(row, col + 2, test_acc)
            worksheet.write(row, col + 3, mean_IOU0)
            row += 1
        workbook.close()

    def save(self, step):
        print('----> Saving the model at step #{0}'.format(step))
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir + self.conf.run_name, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('----> No such checkpoint found', model_path)
            return
        print('----> Restoring the model...')
        self.saver.restore(self.sess, model_path)
        print('----> Model successfully restored')
