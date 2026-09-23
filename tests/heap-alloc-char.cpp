constexpr int x = []() {
  for (unsigned I = 0; I != 10'000; ++I) {
    char *buffer = new char[1024];
    for (unsigned c = 0; c != 1024; ++c)
      buffer[c] = 97 + (c % 26);
    delete[] buffer;
  }
  return 1;
}();
