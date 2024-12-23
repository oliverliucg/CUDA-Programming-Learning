#include "common.h"
#include <stdio.h>

void compareArrays(int* a, int* b, int length) {
  for (int i = 0; i < length; ++i) {
    if (a[i] != b[i]) {
      printf("Arrays differ at index %d: %d != %d\n", i, a[i], b[i]);
      return;
    }
  }
  printf("Arrays match\n");
}

void compareArrays(int* a, int* b, int* c, int length) {
  for (int i = 0; i < length; ++i) {
    if (a[i] != b[i] || a[i] != c[i]) {
      printf("Arrays differ at index %d: %d %d %d\n", i, a[i], b[i], c[i]);
      return;
    }
  }
  printf("Arrays match\n");
}

void printArray(int* a, int length) {
  for (int i = 0; i < length; ++i) {
    printf("%d ", a[i]);
  }
  printf("\n");
}