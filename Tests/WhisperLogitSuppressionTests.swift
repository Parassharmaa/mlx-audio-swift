import MLX
import Testing

@testable import MLXAudioSTT

struct WhisperLogitSuppressionTests {
    @Test func cachedMasksPreserveLogitsAndGreedyTokens() {
        for dtype in [DType.float32, .float16] {
            for timestampBegin in [5, 12, 15] {
                for begin in [[Int](), [-1, 0, 4, 4, 12, 99]] {
                    for suppress in [[Int](), [-5, 1, 4, 5, 99]] {
                        let prepared = WhisperLogitSuppression(
                            vocabularySize: 12,
                            dtype: dtype,
                            beginSuppress: begin,
                            suppress: suppress,
                            timestampBegin: timestampBegin
                        )
                        for step in 0..<5 {
                            let logits = MLXArray(
                                (0..<12).map { Float(($0 + step * 7) % 12) }
                            ).asType(dtype)
                            var expected = logits
                            func oldMask(_ ids: [Int]) -> MLXArray {
                                var values = [Float](repeating: 0, count: 12)
                                for id in ids where id >= 0 && id < 12 {
                                    values[id] = -1e9
                                }
                                return MLXArray(values).asType(dtype)
                            }
                            if step == 0, !begin.isEmpty {
                                expected = expected + oldMask(begin)
                            }
                            if !suppress.isEmpty {
                                expected = expected + oldMask(suppress)
                            }
                            if timestampBegin < 12 {
                                expected = expected + oldMask(Array(timestampBegin..<12))
                            }
                            let actual = prepared.apply(to: logits, firstToken: step == 0)
                            #expect(actual.asArray(Float.self) == expected.asArray(Float.self))
                            #expect(actual.argMax().item(Int.self) == expected.argMax().item(Int.self))
                        }
                    }
                }
            }
        }
    }
}
