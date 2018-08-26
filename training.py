import tensorflow as tf
import gpflow
import numpy as np


def initialize_training(model, objective, ARGS):

    # Create session and initialize model vars
    session = tf.Session()
    model.initialize(session=session, force=False)

    # Create training step
    optimizer = tf.train.AdamOptimizer(ARGS.learning_rate)
    model_vars = model.trainable_tensors
    model_task_vars = [var for var in model_vars if "H" in var.name]
    train_step = optimizer.minimize(objective, var_list=model_vars)

    model_variables = tf.trainable_variables()
    optimizer_slots = [
        optimizer.get_slot(var, name)
        for name in optimizer.get_slot_names()
        for var in model_variables]
    if isinstance(optimizer, tf.train.AdamOptimizer):
        optimizer_slots.extend([
            a for a in optimizer._get_beta_accumulators()
        ])
    optimizer_slots = [var for var in optimizer_slots if var is not None]
    session.run([tf.initialize_variables(optimizer_slots)])

    saver = tf.train.Saver()

    return session, optimizer, train_step, saver

def train(model, objective, data, ARGS):

    session, optimizer, train_step, saver =\
        initialize_training(model, objective, ARGS)

    num_batches = int(ARGS.n_train_tasks / ARGS.batch_size)
    seq = np.arange(ARGS.n_train_tasks)
    for epoch in range(ARGS.train_steps):
        all_obj = []
        np.random.shuffle(seq)
        for b in range(int(num_batches)):
            si = b * ARGS.batch_size
            ei = si + ARGS.batch_size

            X_b, Y_b, ids_b, num_steps, task_scale, data_scale = data.get_batch(seq, si, ei)

            _, obj = session.run(
                [train_step, objective],
                feed_dict={
                    model.X_ph: X_b,
                    model.Y_ph: Y_b,
                    model.H_ids_ph: ids_b,
                    model.num_steps: num_steps,
                    model.data_scale: data_scale,
                    model.task_scale: task_scale})

            all_obj.append(obj)

        mobj = np.mean(all_obj)
        print("Epoch {}/{} :: {:.2f}".format(epoch+1, ARGS.train_steps, mobj))

    saver.save(session, ARGS.model_path)

    return session, saver

def initialize_inference(model, objective, session, data, ARGS):

    # Create training step
    optimizer = tf.train.AdamOptimizer(ARGS.learning_rate)

    model_vars = model.trainable_tensors
    model_task_vars = [var for var in model_vars if "H" in var.name]
    test_ids = data.data["test"]["ids"]
    test_steps = []
    for t in test_ids:
        test_steps.append(optimizer.minimize(
            objective, var_list=[
                var for var in model_task_vars if "H_{}".format(t) in var.name]))

    model_variables = tf.trainable_variables()
    optimizer_slots = [
        optimizer.get_slot(var, name)
        for name in optimizer.get_slot_names()
        for var in model_variables]
    if isinstance(optimizer, tf.train.AdamOptimizer):
        optimizer_slots.extend([
            a for a in optimizer._get_beta_accumulators()
        ])
    optimizer_slots = [var for var in optimizer_slots if var is not None]
    session.run([tf.initialize_variables(optimizer_slots)])

    return optimizer, test_steps

def infer_latent(model, objective, session, saver, data, ARGS):

    optimizer, test_steps =\
        initialize_inference(model, objective, session, data, ARGS)

    for task, infer_step in enumerate(test_steps):
        for step in range(ARGS.infer_steps):
            X_b, Y_b, task_scale, data_scale =\
                data.get_task(task, ARGS.n_test_data, set="test")
            ids_b = np.int32([data.data["test"]["ids"][task]])

            _, obj = session.run(
                [infer_step, objective],
                feed_dict={
                    model.X_ph: X_b,
                    model.Y_ph: Y_b,
                    model.H_ids_ph: ids_b,
                    model.num_steps: ARGS.n_test_data,
                    model.data_scale: data_scale,
                    model.task_scale: task_scale})

            print("Step {}/{} :: {:.2f}".format(step+1, ARGS.infer_steps, obj))

        #model_vars = model.trainable_tensors
        #model_task_vars = [var for var in model_vars if "H" in var.name]
        #print(model_task_vars)

    saver.save(session, ARGS.model_path)

    return session, saver

def predict(model, session, X, ids, use_var=True):

    n_tasks = X.shape[0]
    n_data = X.shape[1]
    dim_in = X.shape[2]

    assert n_tasks == ids.shape[0]

    X = X.reshape(-1, dim_in)

    ymu_pred, yvar_pred, hmu, hvar = model.predict_y(use_var=use_var)
    fd = {
        model.X_ph: X,
        model.X_var_ph: np.zeros_like(X),
        model.H_ids_ph: ids,
        model.num_steps: n_data}
    ymu, yvar, hmu, hvar = session.run([ymu_pred, yvar_pred, hmu, hvar], feed_dict=fd)
    ymu = ymu.reshape(n_tasks, n_data, ymu.shape[1])
    yvar = yvar.reshape(n_tasks, n_data, yvar.shape[1])

    return ymu, yvar, hmu, hvar
