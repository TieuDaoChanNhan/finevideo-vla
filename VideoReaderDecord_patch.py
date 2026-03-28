class VideoReaderDecord:
    """
    Memory-safe video reader with random frame access.

    Unlike the original PyAV helper that decodes the whole video into RAM, this
    class keeps a Decord VideoReader handle open and fetches only the frames
    needed for each clip.
    """

    def __init__(self, path: str, num_threads: int = 1, ctx: Optional[Any] = None) -> None:
        self.path = path
        self.ctx = ctx if ctx is not None else cpu(0)
        self.num_threads = int(num_threads)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video not found: {path}")
        self.reader = VideoReader(path, ctx=self.ctx, num_threads=self.num_threads)
        self.num_frames = len(self.reader)
        if self.num_frames <= 0:
            raise RuntimeError(f"No frames available in video: {path}")

    def get_frames(self, indices: Sequence[int]) -> np.ndarray:
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            raise ValueError("indices must be non-empty")
        if idx.min() < 0 or idx.max() >= self.num_frames:
            raise IndexError(
                f"Requested frame indices out of range for {self.path}. "
                f"min={idx.min()}, max={idx.max()}, num_frames={self.num_frames}"
            )
        return self.reader.get_batch(idx.tolist()).asnumpy()
