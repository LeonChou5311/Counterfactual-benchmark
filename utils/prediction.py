from enum import Enum
import numpy as np

class PredictionType(Enum):
  TruePositive = "TruePositive"
  TrueNegative = "TrueNegative"
  FalsePositive = "FalsePositive"
  FalseNegative = "FalseNegative"

def wrap_information( local_data_dict ):
    
    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []
    for instance in local_data_dict:
        
        # wrap up true positives
        if( instance['prediction_type'] == 'TRUE POSITIVE'):
            true_positives.append(instance)

        # wrap up true negatives
        if( instance['prediction_type'] == 'TRUE NEGATIVE' ):
            true_negatives.append(instance)
        
        # wrap up false positives
        if( instance['prediction_type'] == 'FALSE POSITIVE' ):
            false_positives.append(instance)
        
        # wrap up false negatives
        if( instance['prediction_type'] == 'FALSE NEGATIVE' ):
            false_negatives.append(instance)
            
    return true_positives, true_negatives, false_positives, false_negatives

def generate_local_predictions( X, Y, model, scaler, encoder ):
    
    # get original vector
    orig_vec = np.round(scaler.inverse_transform(X),6)

    # generate all predictions for X
    predictions = model.predict( X )

    # extrace the label of the prediction of X[indx]
    prediction_class = encoder.inverse_transform( predictions )
    local_data_dict = []
    for indx in range(0, orig_vec.shape[0]):

        ground_truth = np.expand_dims(Y[indx], axis=0)
        ground_truth_class = encoder.inverse_transform( ground_truth )[0][0]

        prediction = prediction_class[indx][0]

        # check if data point is a true positive
        if( ( int(prediction) == int(ground_truth_class) ) & (int(prediction)==1) & (int(ground_truth_class)==1) ):
            pred_type = "TRUE POSITIVE"

        # check if data point is a true negative
        if( ( int(prediction) == int(ground_truth_class) ) & (int(prediction)==0) & (int(ground_truth_class)==0) ):
            pred_type = "TRUE NEGATIVE"

        # check if data point is a false negative
        if( ( int(prediction) != int(ground_truth_class) ) & (int(prediction)==0) & (int(ground_truth_class)==1) ):
            pred_type = "FALSE NEGATIVE"

        # check if data point is a false positve
        if( ( int(prediction) != int(ground_truth_class) ) & (int(prediction)==1) & (int(ground_truth_class)==0) ):
            pred_type = "FALSE POSITIVE"

        local_data_dict.append( {'index' : indx,
                                 'original_vector' : orig_vec[indx,:].tolist(),
                                 'scaled_vector' : X[indx,:].tolist(),
                                 'ground_truth' : ground_truth_class,
                                 'predictions' : prediction,
                                 'prediction_type' : pred_type})
    return local_data_dict
  
class PredictionTypeWrapper(object):
  '''
  Class for storing cases in different prediction type

  '''
  def __init__(self, true_positives,true_negatives, false_positives, false_negatives):
    # constructor
    self.true_positives = true_positives
    self.true_negatives = true_negatives
    self.false_positives = false_positives
    self.false_negatives = false_negatives
  def __len__(self,):
    return len(self.true_positives)\
    +len(self.true_negatives)\
    +len(self.false_positives)\
    +len(self.false_negatives)
    
  def true_positives_len(self,):
    return len(self.true_positives)

  def get_len(self, p_t):
    if p_t == PredictionType.TruePositive:
      return self.true_positives_len()
    elif p_t == PredictionType.TrueNegative:
      return self.true_negatives_len()
    elif p_t == PredictionType.FalsePositive:
      return self.false_positives_len()
    elif p_t == PredictionType.FalseNegative:
      return self.false_negatives_len()
    else:
      raise NotImplemented('This prediction type is unsupported.')

  def true_negatives_len(self,):
    return len(self.true_negatives)
  
  def false_negatives_len(self,):
    return len(self.false_negatives)
  
  def false_positives_len(self,):
    return len(self.false_positives)

  def get_instance(self, p_t, index):
    if p_t == PredictionType.TruePositive:
      return self.get_true_positive(index)
    elif p_t == PredictionType.TrueNegative:
      return self.get_true_negative(index)
    elif p_t == PredictionType.FalsePositive:
      return self.get_false_positive(index)
    elif p_t == PredictionType.FalseNegative:
      return self.get_false_nagative(index)
    else:
      raise NotImplemented('This prediction type is unsupported.');

  def get_true_positive(self, index = 0):
    try:
      return self.true_positives[index]
    except:
      raise ValueError("Input index out of range, true positive only have [%d] cases" % (self.true_positives_len()))
  def get_true_negative(self, index = 0):
    try:
      return self.true_negatives[index]
    except:
      raise ValueError("Input index out of range, true negative only have [%d] cases" % (self.true_negatives_len()))
  def get_false_positive(self, index = 0):
    try:
      return self.false_positives[index]
    except:
      raise ValueError("Input index out of range, true positive only have [%d] cases" % (self.false_positives_len()))
  def get_false_nagative(self, index = 0):
    try:
      return self.false_negatives[index]
    except:
      raise ValueError("Input index out of range, true positive only have [%d] cases" % (self.false_negatives_len()))