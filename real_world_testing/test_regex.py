import re

code = """
static int decode_frame(AVCodecContext *avctx,
                        void *data, int *got_frame,
                        AVPacket *avpkt)
{
    return 0;
}
"""

FUNC_REGEX = re.compile(
    r'^[\w\s\*]+?\s+(\w+)\s*\([^)]*\)\s*\{',
    re.MULTILINE
)

print(FUNC_REGEX.findall(code))
