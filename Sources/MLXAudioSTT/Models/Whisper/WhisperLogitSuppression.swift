import MLX

/// Masks are constant for one transcription chunk. Preserve the original
/// addition order, including overlapping suppression IDs and float16 casts.
struct WhisperLogitSuppression {
    private let beginMask: MLXArray?
    private let suppressMask: MLXArray?
    private let timestampMask: MLXArray?

    init(
        vocabularySize: Int,
        dtype: DType,
        beginSuppress: [Int],
        suppress: [Int],
        timestampBegin: Int
    ) {
        func mask(_ ids: [Int]) -> MLXArray? {
            guard !ids.isEmpty else { return nil }
            var values = [Float](repeating: 0, count: vocabularySize)
            for id in ids where id >= 0 && id < vocabularySize {
                values[id] = -1e9
            }
            return MLXArray(values).asType(dtype)
        }
        beginMask = mask(beginSuppress)
        suppressMask = mask(suppress)
        timestampMask = timestampBegin < vocabularySize
            ? mask(Array(timestampBegin..<vocabularySize))
            : nil
    }

    func apply(to logits: MLXArray, firstToken: Bool) -> MLXArray {
        var result = logits
        if firstToken, let beginMask { result = result + beginMask }
        if let suppressMask { result = result + suppressMask }
        if let timestampMask { result = result + timestampMask }
        return result
    }
}
