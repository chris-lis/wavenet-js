import * as tf from '@tensorflow/tfjs';
import { SymbolicTensor } from '@tensorflow/tfjs';

class WaveNet {
  _model: tf.LayersModel;

  constructor(
    channels: number,
    kernelSize = 2,
    dilatationStacks = 5,
    dilatationLevels = 10,
    useLocalConditionning = false,
    useGlobalConditionning = false,
  ) {
    const input = tf.input({shape: [null, channels]})
    const causalConv = tf.layers.conv1d({
      filters: channels,
      kernelSize,
      padding: 'causal'
    }).apply(input)

    let skip = causalConv
    let residual = causalConv

    for (let i = 0; i < dilatationStacks; i++) {
      for (let j = 0; j < dilatationLevels; j++) {
        const dilatedConv = tf.layers.conv1d({
          filters: channels,
          kernelSize,
          dilationRate: 2 ** j,
          padding: 'causal' // ? what should be here
        }).apply(residual);
        const glu = tf.layers.multiply().apply(
        [
          // Here goes global and local conditioning
          tf.layers.activation({activation: 'tanh'}).apply(dilatedConv),
          tf.layers.activation({activation: 'sigmoid'}).apply(dilatedConv),
        ] as SymbolicTensor[])
        skip = tf.layers.add().apply([skip, glu] as SymbolicTensor[])
        residual = glu // Add a hyperparameter here to allow arbitrary residuals
      }
    }

    const outConv1 = tf.layers.conv1d({
      filters: channels,
      kernelSize,
      activation: 'relu',
    }).apply(tf.layers.reLU().apply(skip));
    const outConv2 = tf.layers.conv1d({
      filters: channels,
      kernelSize,
      activation: 'softmax',
    }).apply(outConv1);

    // Compile model
    this._model = tf.model({inputs: input, outputs: outConv2 as SymbolicTensor});

  }
}