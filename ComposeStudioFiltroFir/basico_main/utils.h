int sumAlising(int *a, int *b);

int sumRestrict(int *a, int *b);

int sadd(int a, int b);

void vecsumSinMust(short *restrict sum, short *restrict in1,short *restrict in2, unsigned int N);

void vecsumConMust(short *restrict sum, short *restrict in1,short *restrict in2, unsigned int N);

int dotproducto(short *restrict in1,short *restrict in2, unsigned int N);

int dotproductonassert(short *restrict a, short *restrict b, unsigned int N);

void vecsumSinDesenrrollar(int *restrict sum, int *restrict in1, int *restrict
in2, unsigned int N);

void vecsumDesenrrollado(int *restrict sum, int *restrict in1, int *restrict
in2, unsigned int N);
