#include <cstdio>

int sumAlising(int *a, int *b){
	*b = *b + 10;
	return *a;
}

int sumRestrict(int *restrict a, int *restrict b){
	*b = *b + 10;
	return *a;
}

int sadd(int a, int b)
{
	int result;
	result = a + b;
	if (((a ^ b) & 0x80000000) == 0){
		if ((result ^ a) & 0x80000000){
			result = (a < 0) ? 0x80000000 : 0x7fffffff;
		}
	}
	return (result);
}


void vecsumSinMust(short *restrict sum, short *restrict in1,short *restrict in2, unsigned int N)
{
 int i;
 // uint & _amem4(void *ptr)
 // Permite la carga alineada de 4 bytes de memoria
 for (i = 0; i < N; i++){
	 _amem4(&sum[i]) = _add2(_amem4(&in1[i]), _amem4(&in2[i]));
 }
}

// El MUST ITERATE le dice al compilador el tamaño del bucle
void vecsumConMust(short *restrict sum, short *restrict in1,short *restrict in2, unsigned int N)
{
 int i;
 //Tiene que ser un valor constante
 #pragma MUST_ITERATE (50);
 for (i = 0; i < N; i++){
	 _amem4(&sum[i]) = _add2(_amem4(&in1[i]), _amem4(&in2[i]));
 }
}

int dotproducto(short *restrict in1,short *restrict in2, unsigned int N)
{
	 int i, sum1 = 0, sum2 = 0;
	 for (i = 0; i < N; i++)
	 {
	 sum1 = sum1 + _mpy (_amem4_const(&in1[i]), _amem4_const(&in2[i])); // _mpy muliplica bits menos significativos
	 sum2 = sum2 + _mpyh(_amem4_const(&in1[i]), _amem4_const(&in2[i])); // _mpyh muliplica bits más significativos
	 }
	 return sum1 + sum2;

}

int dotproductonassert(short *restrict a, short *restrict b, unsigned int N)
{
	 int i, sum = 0;
	 /* a and b are aligned to a word boundary */
	 _nassert(((int)(a) & 0x3) == 0);
	 _nassert(((int)(b) & 0x3) == 0);
	 #pragma MUST_ITERATE (40, 40);
	 for (i = 0; i < N; i++)
		 sum += a[i] * b[i];
	 return sum;
}

void vecsumSinDesenrrollar(int *restrict sum, int *restrict in1, int *restrict in2, unsigned int N)
{
	printf("Dentro de sin desenrrollar");
	int i;
	N = 10;
	#pragma MUST_ITERATE (10,10);
	for (i = 0; i < N; i++)
	{
	 sum[i] = _add2(in1[i], in2[i]);
	 printf("%d",i);
	}
}

void vecsumDesenrrollado(int *restrict sum, int *restrict in1, int *restrict in2, unsigned int N)
{
	printf("Dentro de desenrrollado");
	sum[0] = _add2(in1[0], in2[0]);
	sum[1] = _add2(in1[1], in2[1]);
	sum[2] = _add2(in1[2], in2[2]);
	sum[3] = _add2(in1[3], in2[3]);
	sum[4] = _add2(in1[4], in2[4]);
	sum[5] = _add2(in1[5], in2[5]);
	sum[6] = _add2(in1[6], in2[6]);
	sum[7] = _add2(in1[7], in2[7]);
	sum[8] = _add2(in1[8], in2[8]);
	sum[9] = _add2(in1[9], in2[9]);

}

