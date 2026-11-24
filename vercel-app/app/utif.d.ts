declare module "utif" {
  interface TiffIfd {
    width?: number;
    height?: number;
    t274?: number | number[];
  }
  interface UtifLib {
    decode: (buf: ArrayBuffer) => TiffIfd[];
    decodeImage: (buf: ArrayBuffer, ifd: TiffIfd) => void;
    toRGBA8: (ifd: TiffIfd) => Uint8Array;
  }
  const UTIF: UtifLib;
  export default UTIF;
}
