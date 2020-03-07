#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def read_data(file_queue):
    reader=tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    
    defaults = [[0],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0]]
    cvscolunm = tf.io.decode_csv(value, defaults)
    
    featurecolumn=[i for i in cvscolunm[1:-1]]
    labelcolumn = cvscolunm[-1]
    
    return tf.stack(featurecolumn),labelcolumn
def create_pipeline(filename,batch_size,num_epochs=None):
    
    file_queue = tf.train.string_input_producer([filename],num_epochs = num_epochs)
    
    feature, label = read_data(file_queue)
    
    min_after_dequeue = 10000
    capacity = min_after_dequeue + batch_size
    
    feature_batch, label_batch = tf.train.shuffle_batch(
            [feature,label],batch_size=batch_size,capacity=capacity,
            min_after_dequeue = min_after_dequeue)
    return feature_batch, label_batch

x_train_batch, y_train_batch = create_pipeline('/Users/zhuowang/Documents/kaohsiung_OE.csv',32,num_epochs=100)
x_test, y_test = create_pipeline('/Users/zhuowang/Documents/kaohsiung_OE.csv',32)
with tf.Session() as sess:
    
    init_op = tf.global_variables_initializer()
    local_init_op = tf.local_variables_initializer()
    sess.run(init_op)
    sess.run(local_init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    
    try:
        while True:
            if coord.should_stop():
                break
            example, label = sess.run([x_train_batch, y_train_batch])
            print("训练数据：",example)
            print("训练标签：",label)
            
    except tf.errors.OutOfRangeError:
            print('Done reading')
            
    finally:
            coord.request_stop()
            coord.join(threads)
            sess.close()
            

