"""Patch the QuickTime/MP4 video-track tkhd matrix to a 90° display rotation (iPhone portrait style).
Test-fixture helper only. Usage: python tools/mov_rotate.py file.MOV [file2 ...]"""
import struct, sys

def boxes(buf, start, end):
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[i:i + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[i + 8:i + 16])[0]; hdr = 16
        elif size == 0:
            size = end - i
        yield typ.decode("latin1"), i, hdr, size
        i += size

def patch(path):
    buf = bytearray(open(path, "rb").read())
    done = 0
    for typ, i, hdr, size in boxes(buf, 0, len(buf)):
        if typ != "moov":
            continue
        for t2, j, h2, s2 in boxes(buf, i + hdr, i + size):
            if t2 != "trak":
                continue
            tkhd = None; is_video = False
            for t3, k, h3, s3 in boxes(buf, j + h2, j + s2):
                if t3 == "tkhd":
                    tkhd = (k, h3)
                if t3 == "mdia":
                    for t4, m, h4, s4 in boxes(buf, k + h3, k + s3):
                        if t4 == "hdlr" and buf[m + h4 + 8:m + h4 + 12] == b"vide":
                            is_video = True
            if tkhd and is_video:
                k, h3 = tkhd
                ver = buf[k + h3]
                off = k + h3 + 4 + (32 if ver == 1 else 20) + 16
                w = struct.unpack(">I", buf[off + 36:off + 40])[0] >> 16
                hgt = struct.unpack(">I", buf[off + 40:off + 44])[0] >> 16
                # matrix for 90° (displaymatrix rotation -90 in ffprobe): a=0 b=1 u=0 c=-1 d=0 v=0 x=height y=0 w=1
                mat = struct.pack(">9i", 0, 0x10000, 0, -0x10000, 0, 0, hgt << 16, 0, 0x40000000)
                buf[off:off + 36] = mat
                done += 1
    open(path, "wb").write(buf)
    return done

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, patch(p))
