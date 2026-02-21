import tensorflow as tf

model = tf.keras.models.load_model("v4_model_best.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

open("v4_model_tf214.tflite", "wb").write(tflite_model)
