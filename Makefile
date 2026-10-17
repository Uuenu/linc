# linc — build with plain `make`.
#
#   make              static + shared library in build/
#   make test         build and run the test suite
#   make sanitize     run the tests under AddressSanitizer + UBSan
#   make bench        build and run the benchmarks (NATIVE=1 recommended)
#   make examples     build the example programs
#   make install      install into $(PREFIX) (default /usr/local)
#
# Options:
#   NATIVE=1          tune for the build machine (-march/-mcpu=native)
#   BLAS=1            benchmark against the system BLAS too
#                     (Accelerate on macOS, OpenBLAS elsewhere)
#   DEBUG=1           -O0 -g

CC      ?= cc
PREFIX  ?= /usr/local
BUILD   ?= build

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

WARN    := -Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes \
           -Wmissing-prototypes -Wcast-qual -Wpointer-arith -Wvla
CSTD    := -std=c11 -ffp-contract=fast
CFLAGS  ?= $(if $(DEBUG),-O0 -g,-O3)
CPPFLAGS += -Iinclude
ALL_CFLAGS = $(CSTD) $(WARN) $(CFLAGS) $(ARCHFLAGS)
LDLIBS  += -lm

ifeq ($(NATIVE),1)
  ifeq ($(UNAME_M),arm64)
    ARCHFLAGS += -mcpu=native
  else ifeq ($(UNAME_M),aarch64)
    ARCHFLAGS += -mcpu=native
  else
    ARCHFLAGS += -march=native
  endif
endif

ifeq ($(UNAME_S),Darwin)
  SHLIB      := liblinc.dylib
  SHLDFLAGS  := -dynamiclib -install_name @rpath/liblinc.dylib
  BLAS_FLAGS := -DLINC_BENCH_CBLAS -DACCELERATE_NEW_LAPACK -framework Accelerate
else
  SHLIB      := liblinc.so
  SHLDFLAGS  := -shared -Wl,-soname,liblinc.so.0
  BLAS_FLAGS := -DLINC_BENCH_CBLAS -lopenblas
endif

SRCS := $(wildcard src/*.c)
OBJS := $(SRCS:src/%.c=$(BUILD)/obj/%.o)
PIC_OBJS := $(SRCS:src/%.c=$(BUILD)/pic/%.o)
HDRS := include/linc.h src/linc_internal.h
EXAMPLES := $(patsubst examples/%.c,$(BUILD)/%,$(wildcard examples/*.c))

.PHONY: all lib test sanitize bench examples install uninstall clean

all: lib
lib: $(BUILD)/liblinc.a $(BUILD)/$(SHLIB)

$(BUILD)/obj/%.o: src/%.c $(HDRS) | $(BUILD)/obj
	$(CC) $(CPPFLAGS) $(ALL_CFLAGS) -c $< -o $@

$(BUILD)/pic/%.o: src/%.c $(HDRS) | $(BUILD)/pic
	$(CC) $(CPPFLAGS) $(ALL_CFLAGS) -fPIC -c $< -o $@

$(BUILD)/liblinc.a: $(OBJS)
	$(AR) rcs $@ $^

$(BUILD)/$(SHLIB): $(PIC_OBJS)
	$(CC) $(SHLDFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/test_linc: tests/test_linc.c $(BUILD)/liblinc.a
	$(CC) $(CPPFLAGS) $(ALL_CFLAGS) $< $(BUILD)/liblinc.a $(LDFLAGS) -o $@ $(LDLIBS)

test: $(BUILD)/test_linc
	$(BUILD)/test_linc

# Separate build tree so instrumented objects never mix with release ones.
sanitize:
	$(MAKE) BUILD=build-san CFLAGS="-O1 -g -fno-omit-frame-pointer \
	  -fsanitize=address,undefined -fno-sanitize-recover=all" \
	  LDFLAGS="-fsanitize=address,undefined" test

$(BUILD)/bench_linc: bench/bench.c $(BUILD)/liblinc.a
	$(CC) $(CPPFLAGS) $(ALL_CFLAGS) $< $(BUILD)/liblinc.a $(LDFLAGS) -o $@ \
	  $(if $(BLAS),$(BLAS_FLAGS)) $(LDLIBS)

bench: $(BUILD)/bench_linc
	$(BUILD)/bench_linc $(BENCH_ARGS)

$(BUILD)/%: examples/%.c $(BUILD)/liblinc.a
	$(CC) $(CPPFLAGS) $(ALL_CFLAGS) $< $(BUILD)/liblinc.a $(LDFLAGS) -o $@ $(LDLIBS)

examples: $(EXAMPLES)

install: lib
	install -d $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib
	install -m 644 include/linc.h $(DESTDIR)$(PREFIX)/include/
	install -m 644 $(BUILD)/liblinc.a $(DESTDIR)$(PREFIX)/lib/
	install -m 755 $(BUILD)/$(SHLIB) $(DESTDIR)$(PREFIX)/lib/

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/include/linc.h \
	  $(DESTDIR)$(PREFIX)/lib/liblinc.a $(DESTDIR)$(PREFIX)/lib/$(SHLIB)

$(BUILD)/obj $(BUILD)/pic:
	mkdir -p $@

clean:
	rm -rf build build-san
