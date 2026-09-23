
constexpr int add(int a, int b) {
  return 1 + 1 + 1 +  1 + a + 1 + a + 1 + b + 1 + + b;
}

constexpr int mul(int a, int b) {
  return a * a * a * b * b * b;
}


constexpr int run() {
  int Result = 0;
  for (unsigned I = 0; I != 100'000; ++I) {
    Result += add(1,2 );
    Result += mul(2,1);
  }
  return Result;
}
constexpr int R = run();


