# ONNX operator compatibility

| Metric | Value |
| --- | --- |
| ONNX version | unknown (`57b9c6a`) |
| onnx2c revision | `99e97fa` |
| Registered operators | 113/198 (57.1%) |
| Verified operators | 107/198 (54.0%) |
| Passing backend tests | 607/1615 (37.6%) |
| Infrastructure errors | 5 |

* **Registered**: onnx2c dispatches the operator.
* **Verified**: at least one single-operator backend test passes.
* **Passing backend tests**: individual test cases that pass end to end.

## Operator compatibility

Every operator of the pinned ONNX specification's default `ai.onnx` domain, classified from its single-operator tests.

| Operator | Implemented | Tests passed | Status |
| --- | --- | --- | --- |
| AffineGrid | no | 0 | Missing |
| ArgMax | no | 0 | Missing |
| ArgMin | no | 0 | Missing |
| Attention | no | 0 | Missing |
| Bernoulli | no | 0 | Missing |
| BitwiseAnd | no | 0 | Missing |
| BitwiseNot | no | 0 | Missing |
| BitwiseOr | no | 0 | Missing |
| BitwiseXor | no | 0 | Missing |
| BlackmanWindow | no | 0 | Missing |
| CastLike | no | 0 | Missing |
| CenterCropPad | no | 0 | Missing |
| Col2Im | no | 0 | Missing |
| Compress | no | 0 | Missing |
| ConcatFromSequence | no | 0 | Missing |
| CumSum | no | 0 | Missing |
| DFT | no | 0 | Missing |
| DeformConv | no | 0 | Missing |
| DepthToSpace | no | 0 | Missing |
| Det | no | 0 | Missing |
| Einsum | no | 0 | Missing |
| EyeLike | no | 0 | Missing |
| GatherElements | no | 0 | Missing |
| GatherND | no | 0 | Missing |
| Gelu | no | 0 | Missing |
| GlobalLpPool | no | 0 | Missing |
| GridSample | no | 0 | Missing |
| GroupNormalization | no | 0 | Missing |
| HammingWindow | no | 0 | Missing |
| HannWindow | no | 0 | Missing |
| Hardmax | no | 0 | Missing |
| If | no | 0 | Missing |
| ImageDecoder | no | 0 | Missing |
| IsInf | no | 0 | Missing |
| IsNaN | no | 0 | Missing |
| Loop | no | 0 | Missing |
| LpNormalization | no | 0 | Missing |
| LpPool | no | 0 | Missing |
| MaxRoiPool | no | 0 | Missing |
| MaxUnpool | no | 0 | Missing |
| MeanVarianceNormalization | no | 0 | Missing |
| MelWeightMatrix | no | 0 | Missing |
| Mish | no | 0 | Missing |
| Multinomial | no | 0 | Missing |
| NegativeLogLikelihoodLoss | no | 0 | Missing |
| NonMaxSuppression | no | 0 | Missing |
| NonZero | no | 0 | Missing |
| OneHot | no | 0 | Missing |
| Optional | no | 0 | Missing |
| OptionalGetElement | no | 0 | Missing |
| OptionalHasElement | no | 0 | Missing |
| RMSNormalization | no | 0 | Missing |
| RNN | no | 0 | Missing |
| RandomNormal | no | 0 | Missing |
| RandomNormalLike | no | 0 | Missing |
| RandomUniformLike | no | 0 | Missing |
| RegexFullMatch | no | 0 | Missing |
| ReverseSequence | no | 0 | Missing |
| RoiAlign | no | 0 | Missing |
| RotaryEmbedding | no | 0 | Missing |
| STFT | no | 0 | Missing |
| Scan | no | 0 | Missing |
| Scatter | no | 0 | Missing |
| ScatterElements | no | 0 | Missing |
| SequenceAt | no | 0 | Missing |
| SequenceConstruct | no | 0 | Missing |
| SequenceEmpty | no | 0 | Missing |
| SequenceErase | no | 0 | Missing |
| SequenceInsert | no | 0 | Missing |
| SequenceLength | no | 0 | Missing |
| SequenceMap | no | 0 | Missing |
| Size | no | 0 | Missing |
| SoftmaxCrossEntropyLoss | no | 0 | Missing |
| SpaceToDepth | no | 0 | Missing |
| SplitToSequence | no | 0 | Missing |
| StringConcat | no | 0 | Missing |
| StringNormalizer | no | 0 | Missing |
| StringSplit | no | 0 | Missing |
| Swish | no | 0 | Missing |
| TensorScatter | no | 0 | Missing |
| TfIdfVectorizer | no | 0 | Missing |
| Tile | no | 0 | Missing |
| TopK | no | 0 | Missing |
| Trilu | no | 0 | Missing |
| Unique | no | 0 | Missing |
| Pad | yes | 0 | Failing |
| Round | yes | 0 | Failing |
| Split | yes | 0 | Failing |
| AveragePool | yes | 18 | Partial |
| BatchNormalization | yes | 2 | Partial |
| Cast | yes | 8 | Partial |
| Clip | yes | 11 | Partial |
| ConstantOfShape | yes | 2 | Partial |
| Conv | yes | 5 | Partial |
| ConvTranspose | yes | 9 | Partial |
| DequantizeLinear | yes | 4 | Partial |
| Dropout | yes | 6 | Partial |
| Equal | yes | 8 | Partial |
| Identity | yes | 3 | Partial |
| MaxPool | yes | 18 | Partial |
| Mod | yes | 3 | Partial |
| QuantizeLinear | yes | 4 | Partial |
| ReduceL1 | yes | 8 | Partial |
| ReduceL2 | yes | 8 | Partial |
| ReduceLogSum | yes | 4 | Partial |
| ReduceLogSumExp | yes | 8 | Partial |
| ReduceMax | yes | 8 | Partial |
| ReduceMin | yes | 8 | Partial |
| ReduceProd | yes | 8 | Partial |
| ReduceSum | yes | 8 | Partial |
| ReduceSumSquare | yes | 8 | Partial |
| Reshape | yes | 7 | Partial |
| Resize | yes | 11 | Partial |
| ScatterND | yes | 3 | Partial |
| Shape | yes | 2 | Partial |
| Slice | yes | 7 | Partial |
| Max | yes | 13 | Blocked |
| Min | yes | 13 | Blocked |
| RandomUniform | yes | 0 | Untested |
| Abs | yes | 1 | Passing |
| Acos | yes | 2 | Passing |
| Acosh | yes | 2 | Passing |
| Add | yes | 8 | Passing |
| And | yes | 8 | Passing |
| Asin | yes | 2 | Passing |
| Asinh | yes | 2 | Passing |
| Atan | yes | 2 | Passing |
| Atanh | yes | 2 | Passing |
| BitShift | yes | 8 | Passing |
| Ceil | yes | 2 | Passing |
| Celu | yes | 1 | Passing |
| Concat | yes | 12 | Passing |
| Constant | yes | 1 | Passing |
| ConvInteger | yes | 2 | Passing |
| Cos | yes | 2 | Passing |
| Cosh | yes | 2 | Passing |
| Div | yes | 9 | Passing |
| DynamicQuantizeLinear | yes | 3 | Passing |
| Elu | yes | 3 | Passing |
| Erf | yes | 1 | Passing |
| Exp | yes | 2 | Passing |
| Expand | yes | 2 | Passing |
| Flatten | yes | 9 | Passing |
| Floor | yes | 2 | Passing |
| GRU | yes | 4 | Passing |
| Gather | yes | 4 | Passing |
| Gemm | yes | 11 | Passing |
| GlobalAveragePool | yes | 2 | Passing |
| GlobalMaxPool | yes | 2 | Passing |
| Greater | yes | 8 | Passing |
| GreaterOrEqual | yes | 8 | Passing |
| HardSigmoid | yes | 3 | Passing |
| HardSwish | yes | 1 | Passing |
| InstanceNormalization | yes | 2 | Passing |
| LRN | yes | 2 | Passing |
| LSTM | yes | 4 | Passing |
| LayerNormalization | yes | 19 | Passing |
| LeakyRelu | yes | 3 | Passing |
| Less | yes | 8 | Passing |
| LessOrEqual | yes | 8 | Passing |
| Log | yes | 2 | Passing |
| LogSoftmax | yes | 7 | Passing |
| MatMul | yes | 3 | Passing |
| MatMulInteger | yes | 1 | Passing |
| Mean | yes | 3 | Passing |
| Mul | yes | 9 | Passing |
| Neg | yes | 2 | Passing |
| Not | yes | 3 | Passing |
| Or | yes | 8 | Passing |
| PRelu | yes | 2 | Passing |
| Pow | yes | 12 | Passing |
| QLinearConv | yes | 1 | Passing |
| QLinearMatMul | yes | 8 | Passing |
| Range | yes | 2 | Passing |
| Reciprocal | yes | 2 | Passing |
| ReduceMean | yes | 8 | Passing |
| Relu | yes | 1 | Passing |
| Selu | yes | 3 | Passing |
| Shrink | yes | 2 | Passing |
| Sigmoid | yes | 2 | Passing |
| Sign | yes | 1 | Passing |
| Sin | yes | 2 | Passing |
| Sinh | yes | 2 | Passing |
| Softmax | yes | 7 | Passing |
| Softplus | yes | 2 | Passing |
| Softsign | yes | 2 | Passing |
| Sqrt | yes | 2 | Passing |
| Squeeze | yes | 2 | Passing |
| Sub | yes | 9 | Passing |
| Sum | yes | 3 | Passing |
| Tan | yes | 2 | Passing |
| Tanh | yes | 2 | Passing |
| ThresholdedRelu | yes | 3 | Passing |
| Transpose | yes | 7 | Passing |
| Unsqueeze | yes | 7 | Passing |
| Upsample | yes | 1 | Passing |
| Where | yes | 2 | Passing |
| Xor | yes | 8 | Passing |

Status counts: Missing 85, Failing 3, Partial 28, Blocked 2, Untested 1, Passing 79.

Legend: **Missing** – no implementation registered; **Untested** – registered, but no single-operator test exists; **Failing** – registered, tests exist, none pass; **Partial** – at least one test passes, at least one fails; **Blocked** – tests exist, but infrastructure errors leave the result unreliable; **Passing** – all applicable tests pass.

282 failing test cases exercise several operators at once; they are not attributed to any single implementation.

## Failure details

Only operators that are **Failing** or **Blocked**, counted from their single-operator tests.

| Operator | ONNX→C | C compile | Numerical | Runtime | Infrastructure |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pad | 6 | 0 | 0 | 0 | 0 |
| Round | 0 | 0 | 1 | 0 | 0 |
| Split | 16 | 0 | 0 | 0 | 0 |
| Max | 0 | 0 | 0 | 0 | 1 |
| Min | 0 | 0 | 0 | 0 | 1 |

## Coverage limitations

- Passing the available backend tests does not prove full ONNX compatibility.
- Operator versions, attributes, data types, and tensor shapes are not exhaustively tested.
- Multi-operator tests cannot be attributed to a single implementation, so they never change an operator's status.
- Infrastructure errors are tooling limitations, not compiler failures, and are never counted as passing tests.
- Detailed per-case diagnostics live in `operator-coverage.json` and the CI artifacts.
