/*
 * Programa que implementa un filtro FIR
 * con optimizaciones para un DSP Texas Instruments
 *
 *
 * @author Juan Siverio rojas
 */
#include <cstdio>


#define nvMax 3500

float v_datos_default [] = {1,3,2,5,7,11,4,2,1,3};
float v_coef_default [] = {0.33,0.33,0.33};
float salida,valorEntrada,borradorFloat;

int indiceValor,indiceCoeficiente,tmp,nv,nc;
char borradorCaracter;

float * restrict v_coef; ///puntero hacia array con los coeficientes leidos desde fichero csv
float * restrict v_datos; ///puntero hacia array con los datos leidos desde fichero csv

void obtenerValores()
{

	///
	///Primero intento abrir fichero de coeficientes
	///
	FILE *fich_num = fopen("Coeficientes.csv","r");

	if (fich_num == NULL){
		///
		///si no se encontró o pudo abrir, lo indico y utilizao el array por defecto.
		///
		printf("\nNo se puede abrir fichero: Coeficientes.csv");
		printf("\nUsando valores por defecto");

		nc=3;
		v_coef =  v_coef_default;
	}

	else
	{

		nc=0;
		while (!feof(fich_num))
		{
			fscanf(fich_num,"%f",&borradorFloat); ///leo el valor en flotante
			nc++; ///numero de coeficientes que he leido
			if (!feof(fich_num))
				fscanf(fich_num,"%c",&borradorCaracter); ///leo la coma.

		}

		v_coef = (float *)  malloc(nc * sizeof(float));
		fseek(fich_num,0,SEEK_SET); ///me vuelvo a colocar al principio del fichero

		nc=0;
		while (!feof(fich_num))
		{
			fscanf(fich_num,"%f",&v_coef[nc]); ///leo el valor en flotante
			nc++; ///numero de coeficientes que he leido
			if (!feof(fich_num))
				fscanf(fich_num,"%c",&borradorCaracter); ///leo la coma.

		}

		fclose(fich_num);
	}


	///Imprimo los coeficientes leidos.
    for ( tmp = 0; tmp < nc; tmp++)
    	printf("\nCoeficiente %d: %lf",tmp,v_coef[tmp]);

	///
	///Ahora intento lo mismo con fichero de datos
	///
	fich_num = fopen("musica4.csv","r");

	if (fich_num == NULL){
		///
		///si no se encontró o pudo abrir, lo indico y utilizao el array por defecto.
		///
		printf("\nNo se puede abrir fichero: musica4.csv");
		printf("\nUsando valores por defecto");

		nv=10;
		v_datos =  v_datos_default;
	}

	else
	{
 ///aqui leo el fichero y voy contando para saber cuantos memoria tengo que asignar realmente.Así
		//no dependo de un tamaño prefijado, sino depende del tamaño del fichero.
		nv=0;
		while ((!feof(fich_num)) && (nv < nvMax)) ///limite de valores a leer.
		{
			fscanf(fich_num,"%f",&borradorFloat); ///leo el valor en flotante
			nv++; ///numero de coeficientes que he leido
			if (!feof(fich_num))
				fscanf(fich_num,"%c",&borradorCaracter); ///leo la coma.

		}

		v_datos = (float *)  malloc(nv * sizeof(float));
		fseek(fich_num,0,SEEK_SET); ///me vuelvo a colocar al principio del fichero

		nv=0;
		while ((!feof(fich_num)) && (nv < nvMax)) ///establezco un límite de valores en 3500
		{
			fscanf(fich_num,"%f",&v_datos[nv]); ///leo el valor en flotante
			nv++; ///numero de coeficientes que he leido
			if (!feof(fich_num))
				fscanf(fich_num,"%c",&borradorCaracter); ///leo la coma.

		}

		fclose(fich_num);
	}

///Imprimo los valores leidos
	    for ( tmp = 0; tmp < nv; tmp++)
	    	printf("\nValor %d: %lf",tmp,v_datos[tmp]);


}
void filtro_fir()
{
	///Una vez rellenados los vectores de coeficientes y de datos, procedo al cáculo de la salida


		///bucle que comprobará todas las entradas posibles.
	  for (indiceValor = 0; indiceValor < nv; indiceValor++)
	  {
			salida=0.0; ///inicialmente salida = 0

		  //bucle que realiza el cáculo para una entrada dada
		    for (indiceCoeficiente=0; indiceCoeficiente < nc;indiceCoeficiente++) ///tengo que sumar si o sí, mientras no recorra todos los coeficientes.
		    {
		      tmp=indiceValor-indiceCoeficiente;
		    	if ((tmp >= 0) && (tmp < nv))  ///controlo que el índice de datos esté dentro del rango, si no es así, pongo el valor a cero, para que al multiplicar de cero y no aumente nada su valor en la suma.
		    		///esto puede pasar cuando el valor del índice es más pequeño que el número de coeficientes que hay, por ejemplo.

		    		valorEntrada= v_datos[tmp];
		    	else
		    		valorEntrada = 0.0;

		    	salida = salida + (v_coef[indiceCoeficiente]*valorEntrada);

		    }

		 //   printf("\nIteración : %d Valor: %lf",indiceValor,salida);
	  }

}

void filtro_fir_intrinsicos()
{
	///Una vez rellenados los vectores de coeficientes y de datos, procedo al cáculo de la salida

	int borrador;

		///bucle que comprobará todas las entradas posibles.
	  for (indiceValor = 0; indiceValor < nv; indiceValor++)
	  {
			salida=0.0; ///inicialmente salida = 0

		  //bucle que realiza el cáculo para una entrada dada
		    for (indiceCoeficiente=0; indiceCoeficiente < nc;indiceCoeficiente++) ///tengo que sumar si o sí, mientras no recorra todos los coeficientes.
		    {
		      tmp=indiceValor-indiceCoeficiente;

		      	if (_cmpgt2(tmp,-1) && _cmpgt2(nv,tmp)) ///controlo que el índice de datos esté dentro del rango, si no es así, pongo el valor a cero, para que al multiplicar de cero y no aumente nada su valor en la suma.
		    		///esto puede pasar cuando el valor del índice es más pequeño que el número de coeficientes que hay, por ejemplo.

		    		valorEntrada= v_datos[tmp];
		    	else
		    		valorEntrada = 0.0;



		    	salida= _lsadd(_smpy(v_coef[indiceCoeficiente],valorEntrada),salida);


		    }

		    //printf("\nIteración : %d Valor: %lf",indiceValor,salida);
	  }

}
//
//Versión filtro fir con desenrollado de bucle manual
//
void filtro_fir_unroll()
{
	///Una vez rellenados los vectores de coeficientes y de datos, procedo al cáculo de la salida
 #pragma UNROLL(2);

		///bucle que comprobará todas las entradas posibles.
	  for (indiceValor = 0; indiceValor < nv; indiceValor+=2)
	  {
			salida=0.0; ///inicialmente salida = 0

		  //bucle que realiza el cáculo para una entrada dada
		    for (indiceCoeficiente=0; indiceCoeficiente < nc;indiceCoeficiente+=2) ///tengo que sumar si o sí, mientras no recorra todos los coeficientes.
		    {
		      tmp=indiceValor-indiceCoeficiente;
		      if (_cmpgt2(tmp,-1) && _cmpgt2(nv,tmp))  ///controlo que el índice de datos esté dentro del rango, si no es así, pongo el valor a cero, para que al multiplicar de cero y no aumente nada su valor en la suma.
		    		///esto puede pasar cuando el valor del índice es más pequeño que el número de coeficientes que hay, por ejemplo.

		    		valorEntrada= v_datos[tmp];
		    	else
		    		valorEntrada = 0.0;

		      salida= _lsadd(_smpy(v_coef[indiceCoeficiente],valorEntrada),salida);


		      tmp=indiceValor-indiceCoeficiente+1;
			    	if (_cmpgt2(tmp,-1) && _cmpgt2(nv,tmp))  ///controlo que el índice de datos esté dentro del rango, si no es así, pongo el valor a cero, para que al multiplicar de cero y no aumente nada su valor en la suma.
			    		///esto puede pasar cuando el valor del índice es más pequeño que el número de coeficientes que hay, por ejemplo.

			    		valorEntrada= v_datos[tmp];
			    	else
			    		valorEntrada = 0.0;

			    	salida= _lsadd(_smpy(v_coef[indiceCoeficiente+1],valorEntrada),salida);


		    }

		 //   printf("\nIteración : %d Valor: %lf",indiceValor,salida);

		    salida=0.0; ///inicialmente salida = 0

		    		  //bucle que realiza el cáculo para una entrada dada
		    		    for (indiceCoeficiente=0; indiceCoeficiente < nc;indiceCoeficiente+=2) ///tengo que sumar si o sí, mientras no recorra todos los coeficientes.
		    		    {
		    		      tmp=indiceValor+1-indiceCoeficiente;
		    		      if (_cmpgt2(tmp,-1) && _cmpgt2(nv,tmp))  ///controlo que el índice de datos esté dentro del rango, si no es así, pongo el valor a cero, para que al multiplicar de cero y no aumente nada su valor en la suma.
		    		    		///esto puede pasar cuando el valor del índice es más pequeño que el número de coeficientes que hay, por ejemplo.

		    		    		valorEntrada= v_datos[tmp];
		    		    	else
		    		    		valorEntrada = 0.0;

		    		    	salida= _lsadd(_smpy(v_coef[indiceCoeficiente],valorEntrada),salida);

		    			      tmp=indiceValor+1-indiceCoeficiente+1;
		    				    	if (_cmpgt2(tmp,-1) && _cmpgt2(nv,tmp))  ///controlo que el índice de datos esté dentro del rango, si no es así, pongo el valor a cero, para que al multiplicar de cero y no aumente nada su valor en la suma.
		    				    		///esto puede pasar cuando el valor del índice es más pequeño que el número de coeficientes que hay, por ejemplo.

		    				    		valorEntrada= v_datos[tmp];
		    				    	else
		    				    		valorEntrada = 0.0;
		    				    	salida= _lsadd(_smpy(v_coef[indiceCoeficiente+1],valorEntrada),salida);

		    		    }
		//      printf("\nIteración : %d Valor: %lf",indiceValor+1,salida);
	  }

}
void main()
{

obtenerValores();

filtro_fir();
filtro_fir_intrinsicos();
filtro_fir_unroll();


}
