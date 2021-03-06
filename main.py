from DataManager import DataManager
import tensorflow as tf
import numpy as np
import scipy.misc
from fgsm import fgsm
import random
import os
Data = DataManager()

os.environ["CUDA_VISIBLE_DEVICES"]="0"

n_classes = 10
xavier = tf.contrib.layers.xavier_initializer()

adv_steps=100

fgsm_epochs = 9

output = open('results_'+str(fgsm_epochs)+'.txt','w')

for w_mods in [1,2,4,7]:
    for h_mods in [1,2,4,7]:
	tf.reset_default_graph()
        print "h_mods:"+str(h_mods)+" w_mods:"+str(w_mods)
        n_modules = h_mods * w_mods
        def convert(images,h_mods=1,w_mods=1):
            res = []

            h_step=28/h_mods
            w_step=28/w_mods

            for each in images[0]:
                t = np.reshape(each, [28, 28])
                r=[]
                for i in xrange(0,h_mods):
                    for j in xrange(0,w_mods):
                        r.extend(np.reshape(t[i*h_step:(i+1)*h_step,j*w_step:(j+1)*w_step], [-1]).tolist())
                res.append(r)
            return res

        def model(inputs, logits=False):
            x = tf.split(inputs, num_or_size_splits=(n_modules), axis=1)
            W1=[]
            b1=[]

            W2=[]
            b2=[]

            layer1=[]
            layer2=[]

            w1ni=int(np.ceil(float(28*28)/n_modules))
            w1no=int(np.ceil(1000.0/n_modules))
            w2no = int(np.ceil(800.0 / n_modules))

            for i in xrange(0,n_modules):
                W1.append(tf.get_variable("W1_"+str(i+1), [w1ni, w1no], tf.float32, initializer=xavier))
                b1.append(tf.get_variable("b1_"+str(i+1), [w1no], tf.float32))

                W2.append(tf.get_variable("W2_" + str(i + 1), [w1no, w2no], tf.float32, initializer=xavier))
                b2.append(tf.get_variable("b2_" + str(i + 1), [w2no], tf.float32))

                layer1.append(tf.nn.relu(tf.add(tf.matmul(x[i], W1[i]), b1[i], name='bias1_'+str(i+1)), name='relu1_'+str(i+1)))

                layer2.append(tf.nn.relu(tf.add(tf.matmul(layer1[i], W2[i]), b2[i])))



            W3 = tf.get_variable("W3", [w2no*n_modules, 10], tf.float32, initializer=xavier)
            b3 = tf.get_variable("b3", [10], tf.float32)



            layer2s = tf.concat(layer2, 1)

            layer3 = tf.add(tf.matmul(layer2s, W3), b3)

            out = tf.nn.softmax(layer3, name="ybar")

            if logits:
                return out, layer3

            return out

        def _evaluate(x_data, y_data, env):
            #print('\nEvaluating')
            n_sample = x_data.shape[0]
            batch_size = 128
            n_batch = int(np.ceil(n_sample / batch_size))
            loss, acc = 0, 0
            for ind in range(n_batch):
                #print(' batch {0}/{1}'.format(ind + 1, n_batch) + '\n')
                start = ind * batch_size
                end = min(n_sample, start + batch_size)
                batch_loss, batch_acc = sess.run(
                    [env.loss, env.acc],
                    feed_dict={env.x: x_data[start:end],
                               env.y: y_data[start:end]
                               # env.training: False
                               })
                loss += batch_loss * batch_size
                acc += batch_acc * batch_size
            loss /= n_sample
            acc /= n_sample
            #print(' loss: {0:.4f} acc: {1:.4f}'.format(loss, acc))
            return loss, acc


        def _predict(X_data, env):
            #print('\nPredicting')
            n_sample = X_data.shape[0]
            batch_size = 128
            n_batch = int(np.ceil(n_sample / batch_size))
            yval = np.empty((X_data.shape[0], n_classes))
            for ind in range(n_batch):
                #print(' batch {0}/{1}'.format(ind + 1, n_batch) + '\n')
                start = ind * batch_size
                end = min(n_sample, start + batch_size)
                batch_y = sess.run(env.ybar, feed_dict={
                    env.x: X_data[start:end]
                    # , env.training: False
                })
                yval[start:end] = batch_y
            #print()
            return yval
        def export_images(arr,env):

            results=_predict(arr,env)
            h_step = 28 / h_mods
            w_step = 28 / w_mods
            for i in xrange(0,len(arr)):
                narr = np.reshape(arr[i], [-1,28 / h_mods, 28 / w_mods])

                img=np.zeros([28,28])

                for ind_1 in xrange(0,h_mods):
                    for ind_2 in xrange(0,w_mods):
                        img[ind_1*h_step:(ind_1+1)*h_step,ind_2*w_step:(ind_2+1)*w_step]=narr[ind_1*w_mods+ind_2]

                narr = np.reshape(narr, [28, 28],'F')
                #scipy.misc.imsave('data_e6/'+str(i)+'_'+str(np.argmax(results[i]))+'.jpg', img)

        class Dummy():
            pass


        env = Dummy()

        with tf.variable_scope('model'):
            env.x = tf.placeholder(tf.float32, [None, 784], name='x')
            env.y = tf.placeholder(tf.float32, [None, 10], name='y')
            # env.training = tf.placeholder(bool, (), name='mode')

            env.ybar, logits = model(env.x, logits=True)  # training=env.training)

            z = tf.argmax(env.y, axis=1)
            zbar = tf.argmax(env.ybar, axis=1)
            count = tf.cast(tf.equal(z, zbar), tf.float32)
            env.acc = tf.reduce_mean(count, name='acc')

            xent = tf.nn.softmax_cross_entropy_with_logits(labels=env.y,
                                                           logits=logits)
            env.loss = tf.reduce_mean(xent, name='loss')

        env.optim = tf.train.AdamOptimizer(0.001).minimize(env.loss)

        with tf.variable_scope('model', reuse=True):
            env.x_adv = fgsm(model,env.x, epochs=fgsm_epochs, eps=0.01,clip_min=0.0,clip_max=1.0)

        total_loss=0
        total_acc=0
        cumulative_accuracy = 0
        cumulative_loss = 0
        sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)))

        for test_ind in xrange(0,adv_steps):

            if(test_ind % 10 == 0):
                print "model n:"+str(test_ind)+'\n'   

            #sess=tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)))
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            transorm_indexes=range(0,784)
            random.shuffle(transorm_indexes)

            for i in xrange(0, 2001):
                data = Data.get_batch(128,transorm_indexes)
                ins = np.array(convert(data,h_mods,w_mods))
                lbs = data[1]
                sess.run(env.optim, {env.x: ins, env.y: lbs
                                     # ,env.training:True
                                     })
                if (i % 1000 == 0):
                    cl,ca =_evaluate(ins, lbs, env)

            data = Data.get_test_batch(1000, transorm_indexes)
            ins = np.array(convert(data, h_mods, w_mods))
            lbs = data[1]
            cl, ca = _evaluate(ins, lbs, env)

            cumulative_loss+=cl
            cumulative_accuracy+=ca

            data = Data.get_test_batch(512,transorm_indexes)
            X_test = np.array(convert(data,h_mods,w_mods))
            y_test = data[1]

            #print('\nCrafting adversarial')

            n_sample = X_test.shape[0]
            batch_size = 128
            n_batch = int(np.ceil(n_sample/batch_size))
            n_epoch = 20
            X_adv = np.empty_like(X_test)
            for ind in range(n_batch):
                #print(' batch {0}/{1}'.format(ind+1, n_batch)+'\n')
                start = ind*batch_size
                end = min(n_sample, start+batch_size)
                tmp = sess.run(env.x_adv, feed_dict={env.x: X_test[start:end],
                                                         env.y: y_test[start:end]
                                                         #env.training: False
                                                     })
                X_adv[start:end] = tmp
            #print('\nSaving adversarial')
            #os.makedirs('data', exist_ok=True)
            #np.save('data_mod4/ex_00.npy', X_adv)

            #print('\nTesting against adversarial data')
            l,a=_evaluate(X_adv, y_test, env)

            total_loss+=l
            total_acc+=a
            # --------------------------------------------------------------------
            if(test_ind==0):
                export_images(X_adv, env)
            


        print "AVERAGE LOSS: "+str(total_loss/float(adv_steps))+" AVERAGE ACCURACY: "+str(total_acc/float(adv_steps))
        print "MODEL AVERAGE ACC: " +str(cumulative_accuracy/adv_steps) + " MODEL AVERAGE LOSS: "+str(cumulative_loss/adv_steps)
        output.write("H_MODS: "+str(h_mods)+" W_MODS:"+ str(w_mods)+"MODEL AVERAGE ACC: " +str(cumulative_accuracy/adv_steps) + " MODEL AVERAGE LOSS: "+str(cumulative_loss/adv_steps)+"AVERAGE LOSS: "+str(total_loss/float(adv_steps))+" AVERAGE ACCURACY: "+str(total_acc/float(adv_steps))+'\n')
        output.flush()

output.close()
