from Nice4Py import ACL
import numpy as np
import unittest
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import scale
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict



class TestACL(unittest.TestCase):
  def setUp(self):
    a = 1
    # Call this so the mkl libarary is loaded before C++ is called
    pairwise_distances(np.zeros((4,4)), Y=None, metric='euclidean')

  def plot_data(self, ax, label, dim1, dim2):
    markers = ['x', 'o', '^', '*']
    colors = ['darkblue', 'darkred', 'darkgreen', 'darkyellow']
    mks = [markers[int(center)] for center in label]
    cols = [colors[int(center)] for center in label]
    labs = [str(int(center)) for center in label]

    for _x, _y, _mk, _col, _lab in zip(self.data[:, dim1], self.data[:, dim2], mks, cols, labs):
      ax.scatter(_x, _y, marker=_mk, color=_col, label=_lab)

    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

  def plot_comparison(self, ground_truth, pred, dim1, dim2):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    self.plot_data(ax1, ground_truth, dim1, dim2)
    self.plot_data(ax2, pred, dim1, dim2)
    ax1.set_title('Ground Truth')
    ax2.set_title('Prediction')
    plt.savefig('comparison.png', dpi=300)

  def load_data(self, n, d, c, name, type, synthetic=False):
    self.n = n
    self.d = d
    self.c = c
    if synthetic:
      root_dir = '/home/xiangyu/Dropbox/git_project/NICE/cpp/test/data_for_test/kdac/'
    else:
      root_dir = '/home/xiangyu/Dropbox/git_project/NICE/python/test_acl/data/'

    self.sub_name = '_'.join([str(n), str(d), str(c)]) + '.csv'
    if type == 'float':
      self.data_type = np.float32
      data = np.genfromtxt(root_dir + 'data_' + name + '_' + self.sub_name,
                           delimiter=',', dtype=self.data_type)
    elif type == 'double':
      self.data_type = np.float
      data = np.genfromtxt(root_dir + 'data_' + name + '_' + self.sub_name,
                           delimiter=',', dtype=self.data_type)
    else:
      raise "No Such Type " + type
    self.data = data
    return data

  def load_label(self, name, num, synthetic=False):
    if synthetic:
      root_dir = '/home/xiangyu/Dropbox/git_project/NICE/cpp/test/data_for_test/kdac/'
    else:
      root_dir = '/home/xiangyu/Dropbox/git_project/NICE/python/test_acl/data/'
    label = np.genfromtxt(root_dir + 'label' + str(num) + '_' + name + '_' +
                          self.sub_name, delimiter=',', dtype=self.data_type)
    return label

  def load_image_data(self, name):
    im = Image.open('./data/' + name + '.png')

    rgb_im = im.convert('RGB')
    Img_3d_array = np.asarray(rgb_im)
    before_preprocess_data = np.empty((0, 3), dtype=np.uint8)
    data = np.empty((0, 3))
    data_dic = {}
    data_alloc = {}

    for i in range(Img_3d_array.shape[0]):
      for j in range(Img_3d_array.shape[1]):
        data_dic[str(Img_3d_array[i, j])] = Img_3d_array[i, j]

    for i, j in data_dic.items():
      before_preprocess_data = np.vstack((before_preprocess_data, j))
      data = np.vstack((data, j))

    data = scale(data)
    return data




  def label_to_y(self, label):
    N = np.size(label)
    unique_elements = np.unique(label)
    num_of_classes = len(unique_elements)
    class_ids = np.arange(num_of_classes)

    i = 0
    Y = np.zeros(num_of_classes)
    for m in label:
      class_label = np.where(unique_elements == m)[0]
      a_row = np.zeros(num_of_classes)
      a_row[class_label] = 1
      Y = np.hstack((Y, a_row))

    Y = np.reshape(Y, (N + 1, num_of_classes))
    Y = np.delete(Y, 0, 0)

    return Y

  def y_to_label(self, name):
    y = np.genfromtxt('./data/label_' + name + '_' + self.sub_name,
                      delimiter=',', dtype=self.data_type)
    i = 0
    allocation = np.array([])
    for m in range(y.shape[0]):
      allocation = np.hstack((allocation, np.where(y[m] == 1)[0][0]))
      i += 1
    return allocation


  # refer to chieh's test_8.py
  def test_8(self):
    data = self.load_data(n=40, d=2, c=2, name='gaussian', type='float')
    acl = ACL('float', 'ISM', 'cpu')
    acl.set_params(sigma=0.5, c=2, q=1) 
    acl.Fit(data)
    pred = acl.Predict()
    acl.Fit()
    pred_alt = acl.Predict()
    ground_truth = np.concatenate((np.ones(20), np.zeros(20)))

    self.plot_comparison(ground_truth, pred, 0, 1)

    nmi1 = normalized_mutual_info_score(pred, pred_alt)
    nmi2 = normalized_mutual_info_score(ground_truth, pred_alt)

    self.assertAlmostEqual(nmi1, 0.0)
    self.assertAlmostEqual(nmi2, 0.0)

  def test_8_non_vec(self):
    data = self.load_data(n=40, d=2, c=2, name='gaussian', type='float')
    acl = ACL('float', 'cpu', 'ISM')
    acl.set_params(sigma=0.5, c=2, q=1)

    acl.Fit(data)
    pred = acl.Predict()
    acl.Fit()
    pred_alt = acl.Predict()
    ground_truth = np.concatenate((np.ones(20), np.zeros(20)))

    print pred
    print pred_alt

    nmi1 = normalized_mutual_info_score(pred, pred_alt)
    nmi2 = normalized_mutual_info_score(ground_truth, pred_alt)

    self.assertAlmostEqual(nmi1, 0.0)
    self.assertAlmostEqual(nmi2, 0.0)

  def test_8_kdac(self):
    data = self.load_data(n=40, d=2, c=2, name='gaussian', type='float')
    acl = ACL('float', 'cpu', 'KDAC')
    acl.set_params(sigma=0.5, c=2, verbose=1, q=1, debug=0)

    acl.Fit(data)
    pred = acl.Predict()
    print pred

    acl.Fit()
    pred_alt = acl.Predict()
    ground_truth = np.concatenate((np.ones(20), np.zeros(20)))
    print pred_alt

    self.plot_comparison(ground_truth, pred_alt, 0, 1)

    nmi1 = normalized_mutual_info_score(pred, pred_alt)
    nmi2 = normalized_mutual_info_score(ground_truth, pred_alt)
    self.assertAlmostEqual(nmi1, 0.0)
    self.assertAlmostEqual(nmi2, 0.0)

  def test_8_kdac_gpu(self):
    data = self.load_data(n=40, d=2, c=2, name='gaussian', type='float')
    acl = ACL('float', 'gpu')
    acl.set_params(sigma=0.5, c=2, verbose=1, q=1, debug=0)

    acl.Fit(data)
    pred = acl.Predict()
    print pred

    acl.Fit()
    pred_alt = acl.Predict()
    ground_truth = np.concatenate((np.ones(20), np.zeros(20)))
    print pred_alt

    self.plot_comparison(ground_truth, pred_alt, 0, 1)

    nmi1 = normalized_mutual_info_score(pred, pred_alt)
    nmi2 = normalized_mutual_info_score(ground_truth, pred_alt)
    self.assertAlmostEqual(nmi1, 0.0)
    self.assertAlmostEqual(nmi2, 0.0)

  def test_9(self):
    data = self.load_data(n=400, d=4, c=2, name='gaussian', type='float')
    acl = ACL('float', 'cpu', 'ISM')
    acl.set_params(sigma=2.0)
    acl.Fit(data)
    pred = acl.Predict()
    ground_truth = np.concatenate((np.zeros(200), np.ones(200)))
    acl.set_params(sigma=1.0)
    acl.Fit()
    pred_alt = acl.Predict()
    ground_truth_alt = \
      np.concatenate((np.ones(100), np.zeros(100), np.ones(100), np.zeros(100)))

    nmi1 = normalized_mutual_info_score(pred, ground_truth)
    nmi2 = normalized_mutual_info_score(pred_alt, ground_truth_alt)
    self.assertAlmostEqual(nmi1, 1.0)
    self.assertAlmostEqual(nmi2, 1.0)

  def test_9_non_vec(self):
    data = self.load_data(n=400, d=4, c=2, name='gaussian', type='float')
    acl = ACL('float', 'cpu', 'ISM')
    acl.set_params(sigma=2.0)
    acl.Fit(data)
    pred = acl.Predict()
    ground_truth = np.concatenate((np.zeros(200), np.ones(200)))
    acl.set_params(sigma=1.0)
    acl.Fit()
    pred_alt = acl.Predict()
    ground_truth_alt = \
      np.concatenate((np.ones(100), np.zeros(100), np.ones(100), np.zeros(100)))

    nmi1 = normalized_mutual_info_score(pred, ground_truth)
    nmi2 = normalized_mutual_info_score(pred_alt, ground_truth_alt)
    self.assertAlmostEqual(nmi1, 1.0)
    self.assertAlmostEqual(nmi2, 1.0)


  def test_9_kdac(self):
    data = self.load_data(n=400, d=4, c=2, name='gaussian', type='float')
    acl = ACL('float', 'cpu', 'KDAC')
    acl.set_params(debug=0.0, sigma=2.0, verbose=1.0)
    acl.Fit(data)
    pred = acl.Predict()
    ground_truth = np.concatenate((np.zeros(200), np.ones(200)))
    nmi = normalized_mutual_info_score(pred, ground_truth)
    self.assertEqual(nmi, 1.0)

  def test_10(self):
    data = self.load_data(n=164, d=4, c=2, name='moon', type='double')
    acl = ACL('double', 'cpu', 'ISM')
    acl.set_params(c=2, q=2, debug=0, sigma=2, verbose=1)
    acl.Fit(data)
    pred1 = acl.Predict()
    label1 = self.y_to_label('moon1')
    nmi1 = normalized_mutual_info_score(pred1, label1)
    self.assertEqual(nmi1, 1.0)

    acl.set_params(sigma=0.3)
    acl.Fit()
    pred2 = acl.Predict()
    label2 = self.y_to_label('moon2')
    nmi2 = normalized_mutual_info_score(pred2, label2)
    self.assertEqual(nmi2, 1.0)

  def test_10_non_vec(self):
    data = self.load_data(n=164, d=4, c=2, name='moon', type='double')
    acl = ACL('double', 'cpu', 'ISM')
    acl.set_params(c=2, q=2, debug=0, sigma=2, verbose=1)
    acl.Fit(data)
    pred1 = acl.Predict()
    label1 = self.y_to_label('moon1')
    nmi1 = normalized_mutual_info_score(pred1, label1)
    self.assertEqual(nmi1, 1.0)

    acl.set_params(sigma=0.3)
    acl.Fit()
    pred2 = acl.Predict()
    label2 = self.y_to_label('moon2')
    nmi2 = normalized_mutual_info_score(pred2, label2)
    self.assertEqual(nmi2, 1.0)

  def test_10_kdac(self):
    data = self.load_data(n=164, d=4, c=2, name='moon', type='double')
    acl = ACL('double', 'cpu', 'KDAC')
    acl.set_params(c=2, q=2, debug=0, sigma=2, verbose=1, max_time=10)
    acl.Fit(data)
    pred1 = acl.Predict()
    label1 = self.y_to_label('moon1')
    nmi1 = normalized_mutual_info_score(pred1, label1)
    print pred1
    self.assertEqual(nmi1, 1.0)


    acl.set_params(sigma=0.3)
    acl.Fit()
    pred2 = acl.Predict()
    label2 = self.y_to_label('moon2')
    nmi2 = normalized_mutual_info_score(pred2, label2)
    print pred2
    self.assertEqual(nmi2, 1.0)


  def test_11(self):
    data = self.load_data(n=683, d=9, c=2, name='breast', type='double')
    acl = ACL('double', 'cpu', 'ISM')
    acl.set_params(c=2, q=2, sigma=6, verbose=1, debug=0)
    acl.Fit(data)
    pred1 = acl.Predict()
    label = self.load_label('breast', 1)

    acl.Fit()
    pred2 = acl.Predict()
    nmi1 = normalized_mutual_info_score(pred2, label)
    nmi2 = normalized_mutual_info_score(pred1, pred2)

    self.assertAlmostEqual(nmi1, 2.97609971442e-05)
    self.assertAlmostEqual(nmi2, 0.000289983682863)

  def test_11_non_vec(self):
    data = self.load_data(n=683, d=9, c=2, name='breast', type='double')
    acl = ACL('double', 'cpu', 'ISM')
    acl.set_params(c=2, q=2, sigma=6, verbose=1, debug=0)
    acl.Fit(data)
    pred1 = acl.Predict()
    label = self.load_label('breast', 1)

    acl.Fit()
    pred2 = acl.Predict()
    nmi1 = normalized_mutual_info_score(pred2, label)
    nmi2 = normalized_mutual_info_score(pred1, pred2)

    self.assertAlmostEqual(nmi1, 2.97609971442e-05)
    self.assertAlmostEqual(nmi2, 0.000289983682863)

  def test_11_kdac(self):
    data = self.load_data(n=683, d=9, c=2, name='breast', type='double')
    acl = ACL('double', 'cpu', 'KDAC')
    acl.set_params(c=2, q=2, sigma=6, verbose=1, debug=0)
    acl.Fit(data)
    pred1 = acl.Predict()
    label = self.load_label('breast', 1)

    acl.Fit()
    pred2 = acl.Predict()
    nmi1 = normalized_mutual_info_score(pred2, label)
    nmi2 = normalized_mutual_info_score(pred1, pred2)

    self.assertAlmostEqual(nmi1, 2.97609971442e-05)
    self.assertAlmostEqual(nmi2, 0.000289983682863)

  def test_12(self):
    data = self.load_data(n=624, d=27, c=4, name='facial', type='double')
    acl = ACL('double', 'cpu', 'ISM')
    label_identity = self.load_label('facial', 1)
    label_pose = self.load_label('facial', 2)
    y_identity = self.label_to_y(label_identity)
    d_matrix = pairwise_distances(data, Y=None, metric='euclidean')
    acl.set_params(c=4, sigma=np.median(d_matrix), q=4,
                   Lambda=0.1, debug=0, verbose=1)
    acl.Fit(data, y_identity)
    pred = acl.Predict()
    against_identity = normalized_mutual_info_score(pred, label_identity)
    against_pose = normalized_mutual_info_score(pred, label_pose)
    print 'Alternative against identity: ', against_identity
    print 'Alternative against pose:', against_pose

  def test_12_1(self):
    data = self.load_data(n=624, d=27, c=4, name='facial', type='float')
    acl = ACL('float', 'cpu', 'ISM')
    label_identity = self.load_label('facial', 1)
    label_pose = self.load_label('facial', 2)
    y_identity = self.label_to_y(label_identity)
    d_matrix = pairwise_distances(data, Y=None, metric='euclidean')
    acl.set_params(c=4)
    acl.set_params(sigma=np.median(d_matrix).astype(np.float64))
    acl.set_params(q=4)
    acl.set_params(Lambda=0.1)
    acl.set_params(debug=0)
    acl.set_params(verbose=1)
    acl.Fit(data)
    pred = acl.Predict()
    against_identity = normalized_mutual_info_score(pred, label_identity)
    against_pose = normalized_mutual_info_score(pred, label_pose)
    print 'Alternative against identity: ', against_identity
    print 'Alternative against pose:', against_pose

  def test_12_kdac(self):
    data = self.load_data(n=624, d=27, c=4, name='facial', type='double')
    acl = ACL('double', 'cpu', 'KDAC')
    label_identity = self.load_label('facial', 1)
    label_pose = self.load_label('facial', 2)
    y_identity = self.label_to_y(label_identity)
    d_matrix = pairwise_distances(data, Y=None, metric='euclidean')
    acl.set_params(c=4, sigma=np.median(d_matrix), q=4,
                   Lambda=0.1, debug=0, verbose=1)
    acl.Fit(data, y_identity)
    pred = acl.Predict()
    against_identity = normalized_mutual_info_score(pred, label_identity)
    against_pose = normalized_mutual_info_score(pred, label_pose)
    print 'Alternative against identity: ', against_identity
    print 'Alternative against pose:', against_pose

  def test_13(self):
    data = self.load_data(n=164, d=7, c=2, name='moon', type='double')
    data = scale(data)
    data[:, 6] = data[:, 6] / 3.0
    data[:, 5] = data[:, 5] / 3.0
    data[:, 4] = data[:, 4] / 3.0

    acl = ACL('double', 'cpu', 'ISM')
    acl.set_params(c=2, sigma=2, Lambda=0.02, debug=0, verbose=1)
    acl.Fit(data)
    pred1 = acl.Predict()
    label1 = self.y_to_label('moon1')

    acl.set_params(q=2, sigma=0.1)
    acl.Fit()
    pred2 = acl.Predict()
    label2 = self.y_to_label('moon2')

    nmi1 = normalized_mutual_info_score(pred1, label1)
    nmi2 = normalized_mutual_info_score(pred2, label2)
    nmi3 = normalized_mutual_info_score(pred1, pred2)
    self.assertAlmostEqual(nmi1, 1.0)
    self.assertAlmostEqual(nmi2, 1.0)
    self.assertAlmostEqual(nmi3, 0.0)

  def test_13_kdac(self):
    data = self.load_data(n=164, d=7, c=2, name='moon', type='double')
    data = scale(data)
    data[:, 6] = data[:, 6] / 3.0
    data[:, 5] = data[:, 5] / 3.0
    data[:, 4] = data[:, 4] / 3.0

    acl = ACL('double', 'cpu', 'KDAC')
    acl.set_params(c=2, sigma=2, Lambda=0.02, debug=0, verbose=1)
    acl.Fit(data)
    pred1 = acl.Predict()
    label1 = self.y_to_label('moon1')


    acl.set_params(q=2, sigma=0.1)
    acl.Fit()
    pred2 = acl.Predict()
    label2 = self.y_to_label('moon2')

    nmi1 = normalized_mutual_info_score(pred1, label1)
    nmi2 = normalized_mutual_info_score(pred2, label2)
    nmi3 = normalized_mutual_info_score(pred1, pred2)
    self.assertAlmostEqual(nmi1, 1.0)
    self.assertAlmostEqual(nmi2, 1.0)
    self.assertAlmostEqual(nmi3, 0.0)

  def cpu_270_100_3_ism(self):
    data = self.load_data(n=270, d=100, c=3, name='gaussian',
                          type='float', synthetic=True)
    # data = scale(data)
    acl = ACL('float', 'cpu', 'ISM')
    acl.set_params(verbose=1, debug=0, sigma=1.0, q=3, c=3)
    label1 = self.load_label('gaussian', 1, True)
    label2 = self.load_label('gaussian', 2, True)
    y1= self.label_to_y(label1)
    acl.Fit(data, y1)
    pred = acl.Predict()
    nmi = normalized_mutual_info_score(pred, label2)

    self.plot_comparison(label2, pred, 2, 3)

  def cpu_300_100_3_ism(self):
    data = self.load_data(n=300, d=100, c=3, name='gaussian',
                          type='float', synthetic=True)
    # data = scale(data)
    acl = ACL('float', 'cpu', 'ISM')
    acl.set_params(verbose=1, debug=0, sigma=1.0, q=3, c=3)
    label1 = self.load_label('gaussian', 1, True)
    label2 = self.load_label('gaussian', 2, True)
    y1= self.label_to_y(label1)
    acl.Fit(data, y1)
    pred = acl.Predict()
    nmi = normalized_mutual_info_score(pred, label2)

    self.plot_comparison(label2, pred, 2, 3)

  def cpu_300_6_3_ism(self):
    data = self.load_data(n=300, d=6, c=3, name='gaussian',
                          type='float', synthetic=True)
    # data = scale(data)
    acl = ACL('float', 'cpu', 'ISM')
    acl.set_params(verbose=1, debug=0, sigma=1.0, q=3, c=3)
    label1 = self.load_label('gaussian', 1, True)
    label2 = self.load_label('gaussian', 2, True)
    y1= self.label_to_y(label1)
    acl.Fit(data, y1)
    pred = acl.Predict()
    nmi = normalized_mutual_info_score(pred, label2)

    self.plot_comparison(label2, pred, 2, 3)


  def cpu_300_6_3_kdac(self):
    data = self.load_data(n=300, d=6, c=3, name='gaussian',
                          type='float', synthetic=True)
    # data = scale(data)
    acl = ACL('float', 'cpu', 'KDAC')
    acl.set_params(verbose=1, debug=0, sigma=1.0, q=3, c=3)
    label1 = self.load_label('gaussian', 1, True)
    label2 = self.load_label('gaussian', 2, True)
    y1= self.label_to_y(label1)
    acl.Fit(data, y1)
    pred = acl.Predict()
    nmi = normalized_mutual_info_score(pred, label2)

    self.plot_comparison(label2, pred, 2, 3)

  def cpu_30_6_3_ism(self):
    data = self.load_data(n=30, d=6, c=3, name='gaussian',
                          type='float', synthetic=True)
    # data = scale(data)
    acl = ACL('float', 'cpu', 'ISM')
    acl.set_params(verbose=1, debug=0, sigma=1.0, q=3, c=3)
    label1 = self.load_label('gaussian', 1, True)
    label2 = self.load_label('gaussian', 2, True)
    y1= self.label_to_y(label1)
    acl.Fit(scale(data), y1)
    pred = acl.Predict()
    nmi = normalized_mutual_info_score(pred, label2)
    self.plot_comparison(label2, pred, 2, 3)
    print pred
    print nmi

  def cpu_30_6_3_kdac(self):
    data = self.load_data(n=30, d=6, c=3, name='gaussian',
                          type='float', synthetic=True)
    # data = scale(data)
    acl = ACL('float', 'cpu', 'KDAC')
    acl.set_params(verbose=1, debug=0, sigma=1.0, q=3, c=3)
    label1 = self.load_label('gaussian', 1, True)
    label2 = self.load_label('gaussian', 2, True)
    y1= self.label_to_y(label1)
    acl.Fit(scale(data), y1)
    pred = acl.Predict()
    nmi = normalized_mutual_info_score(pred, label2)
    self.plot_comparison(label2, pred, 2, 3)
    print pred
    print nmi

  def gpu_30_6_3_kdac(self):
    data = self.load_data(n=30, d=6, c=3, name='gaussian',
                          type='float', synthetic=True)
    # data = scale(data)
    acl = ACL('float', 'gpu', 'KDAC')
    acl.set_params(verbose=1, debug=0, sigma=1.0, q=3, c=3)
    label1 = self.load_label('gaussian', 1, True)
    label2 = self.load_label('gaussian', 2, True)
    y1= self.label_to_y(label1)
    acl.Fit(data, y1)
    pred = acl.Predict()
    nmi = normalized_mutual_info_score(pred, label2)
    self.plot_comparison(label2, pred, 2, 3)
    print pred
    print nmi

  # def test_14(self):
  #   data = self.load_data(n=1040, d=139, c=4, name='webkb', type='double')
  #   label_univ = self.readLabel('univ')
  #   label_topic = self.readLabel('topic')
  #   y = self.labelToY(label_univ)
  #   acl = ACL('double', 'cpu')
  #   d_matrix = pairwise_distances(data, Y=None, metric='euclidean')
  #   acl.set_params(q=4, c=4, sigma=np.median(d_matrix), Lambda=0.057)
  #   acl.set_params(debug=0, verbose=1)
  #   acl.Fit(data, y)
  #   pred = acl.Predict()
  #   against_alter = normalized_mutual_info_score(label_univ, pred)
  #   against_truth = normalized_mutual_info_score(label_topic, pred)
  #   print against_alter#0.00956
  #   print against_truth#0.3675
  #
  # def test_14_kdac(self):
  #   data = self.load_data(n=1040, d=139, c=4, name='webkb', type='double')
  #   label_univ = self.readLabel('univ')
  #   label_topic = self.readLabel('topic')
  #   y = self.labelToY(label_univ)
  #   acl = ACL('double', 'cpu')
  #   d_matrix = pairwise_distances(data, Y=None, metric='euclidean')
  #   acl.set_params(q=4, c=4, sigma=np.median(d_matrix), Lambda=0.057)
  #   acl.set_params(debug=0, verbose=1, method='KDAC')
  #   acl.Fit(data, y)
  #   pred = acl.Predict()
  #   against_alter = normalized_mutual_info_score(label_univ, pred)
  #   against_truth = normalized_mutual_info_score(label_topic, pred)
  #   print against_alter#0.00956
  #   print against_truth#0.3675
  #
  # def test_15(self):
  #   data = self.load_image_data('Flower3')
  #   print data.dtype
  #   print data.shape
  #
  #   acl = ACL('double', 'cpu')
  #   d_matrix = pairwise_distances(data, Y=None, metric='euclidean')
  #   sigma = 0.1*np.median(d_matrix)
  #   acl.set_params(c=2, q=2, sigma=sigma)
  #   acl.Fit(data)
  #   pred = acl.Predict()


if __name__ == '__main__':
  unittest.main()
