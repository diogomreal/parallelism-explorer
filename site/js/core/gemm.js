// GEMM efficiency: a ceiling times tile quantization on the token (M) dimension. Rows are padded up to whole tiles of A.gemmTileM
// (64 = the smallest single-CTA UMMA tile on Blackwell). Small-M GEMMs are HBM-bound anyway, so the padding matters where GEMMs turn
// compute-bound: prefill and grouped expert GEMMs whose rows per expert sit near a tile boundary.
export const gemmEta = (rows, ceil, A) => (rows <= 0 ? ceil : ceil * rows / (Math.ceil(rows / A.gemmTileM) * A.gemmTileM));
