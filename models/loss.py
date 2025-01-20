import tensorflow as tf

# define custom mse loss (with flooding)
def MC_mse(y_true, y_pred):
	# MSE loss on CFD field data
	squared_error = tf.square(y_pred - y_true)
	sum_squared_error = tf.math.reduce_sum(squared_error)
	mse_loss = sum_squared_error / tf.size(y_true, out_type=tf.dtypes.float32)
	# mse_loss = tf.math.abs(mse_loss - 0.2) + 0.2	# flooding

	# Mass conservation loss on prediction UX field data
	# rho*dY*sum(Uo - Ui) = 0
	mc_pred = []
	for i in range(y_pred.shape[0]):	# each field data in batch
		sum_U = 0
  		# loop over each in and out Temp data
		for j in range(y_true.shape[1]):
			dU_in = y_true[i][j,0]
			dU_out = y_true[i][j,y_true.shape[2] - 1]
			dU = (dU_out - dU_in)
			sum_U += dU
		mean_dU = sum_U / y_true.shape[1]
		m_pred = 1.2041*mean_dU
		mc_pred.append(m_pred)

	mc_true = []
	for i in range(y_true.shape[0]):	# each field data in batch
		sum_U = 0
  		# loop over each in and out Temp data
		for j in range(y_true.shape[1]):
			dU_in = y_true[i][j,0]
			dU_out = y_true[i][j,y_true.shape[2] - 1]
			dU = (dU_out - dU_in)
			sum_U += dU
		mean_dU = sum_U / y_true.shape[1]
		m_true = 1.2041*mean_dU
		mc_true.append(m_true)

	mc_pred = tf.convert_to_tensor(mc_pred, dtype=tf.dtypes.float32)
	mc_true = tf.convert_to_tensor(mc_true, dtype=tf.dtypes.float32)
	mc_error = tf.square(mc_pred - mc_true)
	sum_mc_error = tf.math.reduce_sum(mc_error)
	mc_loss = sum_mc_error / tf.size(mc_true, out_type=tf.dtypes.float32)
	loss = tf.add(mse_loss, mc_loss)
	return loss

def MC_mae(y_true, y_pred):
    # MAE loss on CFD field data
    absolute_error = tf.abs(y_pred - y_true)
    sum_absolute_error = tf.math.reduce_sum(absolute_error)
    mae_loss = sum_absolute_error / tf.size(y_true, out_type=tf.dtypes.float32)
    
    # Mass conservation loss on prediction UX field data
    mc_pred = []
    for i in range(y_pred.shape[0]):  # each field data in batch
        sum_U = 0
        # loop over each in and out Temp data
        for j in range(y_true.shape[1]):
            dU_in = y_true[i][j, 0]
            dU_out = y_true[i][j, y_true.shape[2] - 1]
            dU = (dU_out - dU_in)
            sum_U += dU
        mean_dU = sum_U / y_true.shape[1]
        m_pred = 1.2041 * mean_dU
        mc_pred.append(m_pred)

    mc_true = []
    for i in range(y_true.shape[0]):  # each field data in batch
        sum_U = 0
        # loop over each in and out Temp data
        for j in range(y_true.shape[1]):
            dU_in = y_true[i][j, 0]
            dU_out = y_true[i][j, y_true.shape[2] - 1]
            dU = (dU_out - dU_in)
            sum_U += dU
        mean_dU = sum_U / y_true.shape[1]
        m_true = 1.2041 * mean_dU
        mc_true.append(m_true)

    mc_pred = tf.convert_to_tensor(mc_pred, dtype=tf.dtypes.float32)
    mc_true = tf.convert_to_tensor(mc_true, dtype=tf.dtypes.float32)
    mc_error = tf.abs(mc_pred - mc_true)
    sum_mc_error = tf.math.reduce_sum(mc_error)
    mc_loss = sum_mc_error / tf.size(mc_true, out_type=tf.dtypes.float32)
    
    # Total loss
    loss = tf.add(mae_loss, mc_loss)
    return loss

def MC_huber_loss(y_true, y_pred, delta=1.0):
    # Huber loss on CFD field data
    error = y_pred - y_true
    abs_error = tf.abs(error)
    
    # Huber loss formula
    huber_loss = tf.where(abs_error <= delta, 
                          0.5 * tf.square(error), 
                          delta * (abs_error - 0.5 * delta))
    
    sum_huber_loss = tf.math.reduce_sum(huber_loss)
    huber_loss_mean = sum_huber_loss / tf.size(y_true, out_type=tf.dtypes.float32)
    
    # Mass conservation loss on prediction UX field data
    mc_pred = []
    for i in range(y_pred.shape[0]):  # each field data in batch
        sum_U = 0
        # loop over each in and out Temp data
        for j in range(y_true.shape[1]):
            dU_in = y_true[i][j, 0]
            dU_out = y_true[i][j, y_true.shape[2] - 1]
            dU = (dU_out - dU_in)
            sum_U += dU
        mean_dU = sum_U / y_true.shape[1]
        m_pred = 1.2041 * mean_dU
        mc_pred.append(m_pred)

    mc_true = []
    for i in range(y_true.shape[0]):  # each field data in batch
        sum_U = 0
        # loop over each in and out Temp data
        for j in range(y_true.shape[1]):
            dU_in = y_true[i][j, 0]
            dU_out = y_true[i][j, y_true.shape[2] - 1]
            dU = (dU_out - dU_in)
            sum_U += dU
        mean_dU = sum_U / y_true.shape[1]
        m_true = 1.2041 * mean_dU
        mc_true.append(m_true)

    mc_pred = tf.convert_to_tensor(mc_pred, dtype=tf.dtypes.float32)
    mc_true = tf.convert_to_tensor(mc_true, dtype=tf.dtypes.float32)
    mc_error = tf.abs(mc_pred - mc_true)
    sum_mc_error = tf.math.reduce_sum(mc_error)
    mc_loss = sum_mc_error / tf.size(mc_true, out_type=tf.dtypes.float32)
    
    # Total loss
    loss = tf.add(huber_loss_mean, mc_loss)
    return loss