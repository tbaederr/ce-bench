constexpr char str[] = {
#embed "sqlite3.c" suffix(,0)
};
consteval unsigned checksum(const char *s) {
  unsigned result = 0;
  for (const char *p = s; *p != '\0'; ++p) {
    result += *p;
  }
  return result;
}
constexpr unsigned C = checksum(str);
