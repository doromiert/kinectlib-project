{
  description = "libfreenect2, Protonect, and Python dev env for Kinect v2";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};

      runtimeLibs = with pkgs; [
        stdenv.cc.cc.lib
        libusb1
        glfw3
        libjpeg_turbo
        libGL
        libGLU
        pkg-config
      ];

      libfreenect2 = pkgs.stdenv.mkDerivation rec {
        pname = "libfreenect2";
        version = "0.2.0";

        src = pkgs.fetchFromGitHub {
          owner = "OpenKinect";
          repo = "libfreenect2";
          rev = "v${version}";
          hash = "sha256-5JjZANfkkgK8YRXrdfOCXOBMXxW+UNv6JiNfEDUQscc=";
        };

        nativeBuildInputs = with pkgs; [ cmake pkg-config patchelf ];
        buildInputs = runtimeLibs;

        cmakeFlags = [
          "-DCMAKE_POLICY_VERSION_MINIMUM=3.10"
          "-DBUILD_EXAMPLES=ON"
          "-DENABLE_OPENGL=ON"
          "-DENABLE_OPENCL=OFF"
          "-DENABLE_CUDA=OFF"
        ];

        postInstall = ''
          mkdir -p $out/bin
          cp bin/Protonect $out/bin/Protonect
          patchelf --set-rpath "$out/lib:${pkgs.lib.makeLibraryPath runtimeLibs}" $out/bin/Protonect
          patchelf --set-rpath "$out/lib:${pkgs.lib.makeLibraryPath runtimeLibs}" $out/lib/libfreenect2.so
          mkdir -p $out/lib/udev/rules.d
          cp ../platform/linux/udev/90-kinect2.rules $out/lib/udev/rules.d/90-kinect2.rules
        '';

        meta = with pkgs.lib; {
          description = "Driver for Kinect for Windows v2 / Xbox One";
          homepage    = "https://github.com/OpenKinect/libfreenect2";
          license     = licenses.asl20;
          platforms   = platforms.linux;
        };
      };

      # C shim — wraps libfreenect2 C++ API, exposes plain C symbols for ctypes
      kinect-shim = pkgs.stdenv.mkDerivation {
        pname   = "kinect-shim";
        version = "0.1.0";

        src = ./.; # picks up kinect_shim.cpp from the flake dir

        nativeBuildInputs = with pkgs; [ pkg-config ];
        buildInputs = [ libfreenect2 ] ++ runtimeLibs;

        buildPhase = ''
          g++ -std=c++14 -shared -fPIC -O2 \
            -I${libfreenect2}/include \
            -L${libfreenect2}/lib \
            kinect_shim.cpp \
            -lfreenect2 \
            -o libkinect_shim.so \
            -Wl,-rpath,${libfreenect2}/lib
        '';

        installPhase = ''
          mkdir -p $out/lib
          cp libkinect_shim.so $out/lib/
        '';
      };

      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        numpy
        opencv4
      ]);

    in
    {
      packages.${system} = {
        inherit libfreenect2 kinect-shim;
        default = libfreenect2;
      };

      apps.${system}.default = {
        type    = "app";
        program = "${libfreenect2}/bin/Protonect";
      };

      devShells.${system}.default = pkgs.mkShell {
        name = "kinect-dev";

        packages = [
          pythonEnv
          pkgs.python3Packages.pip
          libfreenect2
          kinect-shim
          pkgs.python3Packages.pygame
          pkgs.python3Packages.flask
          pkgs.python3Packages.insightface
          pkgs.python3Packages.onnxruntime
          pkgs.python3Packages.pyaudio

        ] ++ runtimeLibs;

        shellHook = ''
          export KINECT_SHIM_SO="${kinect-shim}/lib/libkinect_shim.so"
          export LD_LIBRARY_PATH="${libfreenect2}/lib:${kinect-shim}/lib:${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH"

          if [ ! -d .venv ]; then
            python -m venv --system-site-packages .venv
            echo "created .venv — installing mediapipe + faster-whisper..."
            .venv/bin/pip install -q mediapipe faster-whisper
          fi
          source .venv/bin/activate

          echo "kinect dev shell ready"
          echo "shim: $KINECT_SHIM_SO"
        '';
      };
    };
}
