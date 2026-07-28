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
        speexdsp   # nkit/audio/aec.py — ctypes AEC binding needs libspeexdsp.so on LD_LIBRARY_PATH
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

      # nix run .#record-server — phone-controlled gesture capture rig
      # (nkit/record.py). Fully nixpkgs-native, no venv: record.py does no
      # detection, so it needs only cv2 + numpy + websockets, and nkit's
      # __init__ imports lazily so pulling in nkit.record doesn't drag
      # mediapipe (not in nixpkgs) along with it.
      recordPythonEnv = pkgs.python3.withPackages (ps: with ps; [
        numpy
        opencv4
        websockets
      ]);

      record-server = pkgs.writeShellApplication {
        name = "record-server";
        text = ''
          export KINECT_SHIM_SO="${kinect-shim}/lib/libkinect_shim.so"
          export LD_LIBRARY_PATH="${libfreenect2}/lib:${kinect-shim}/lib:${pkgs.lib.makeLibraryPath runtimeLibs}:''${LD_LIBRARY_PATH:-}"

          # Walk up to the repo root instead of trusting $PWD: nix resolves the
          # flake by searching upward, so `nix run .#record-server` works from
          # any subdirectory — but the script's cwd stays where you invoked it,
          # and a bare `.venv` then picks up whatever venv happens to be there
          # (nkit-testapp/ has its own).
          root="$PWD"
          while [ "$root" != "/" ] && [ ! -f "$root/nkit/record.py" ]; do
            root="$(dirname "$root")"
          done
          if [ ! -f "$root/nkit/record.py" ]; then
            echo "record-server: couldn't find the nkit repo above $PWD" >&2
            exit 1
          fi
          cd "$root"

          # relative --out paths resolve against the repo root, not your cwd.
          # cwd is on sys.path for `python -m`, which is how nkit/ is found.
          exec ${recordPythonEnv}/bin/python -u -m nkit.record "$@"
        '';
      };

      # openwakeword isn't in nixpkgs, install via pip into the venv
      # listed here just so the comment is near the shell that uses it
      trainerPythonEnv = pkgs.python3.withPackages (ps: with ps; [
        numpy
        flask
        pyaudio
      ]);

    in
    {
      packages.${system} = {
        inherit libfreenect2 kinect-shim record-server;
        default = libfreenect2;
      };

      apps.${system} = {
        default = {
          type    = "app";
          program = "${libfreenect2}/bin/Protonect";
        };

        # nix run .#record-server -- --out recordings/
        record-server = {
          type    = "app";
          program = "${record-server}/bin/record-server";
        };
      };

      devShells.${system} = {
        default = pkgs.mkShell {
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
            pkgs.nodejs    # nkit-testapp/ (Electron demo UI) — npm/node for deps
            pkgs.electron  # run the app via THIS electron, not the npm-fetched one —
                            # npm's prebuilt Electron binary isn't patched for NixOS's
                            # dynamic linker layout and won't run as-is. see nkit-testapp/README.

          ] ++ runtimeLibs;

          shellHook = ''
            export KINECT_SHIM_SO="${kinect-shim}/lib/libkinect_shim.so"
            export LD_LIBRARY_PATH="${libfreenect2}/lib:${kinect-shim}/lib:${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH"

            if [ ! -d .venv ]; then
              python -m venv --system-site-packages .venv
              echo "created .venv"
            fi
            # always ensure these are present, not just on first .venv creation —
            # an existing .venv from before a dep was added here would otherwise
            # silently never pick it up
            .venv/bin/pip install -q mediapipe faster-whisper openwakeword websockets
            source .venv/bin/activate

            if [ ! -d nkit-testapp/node_modules ]; then
              echo "nkit-testapp/node_modules missing — run 'npm install' inside nkit-testapp/ to fetch Electron"
            fi

            echo "kinect dev shell ready"
            echo "shim: $KINECT_SHIM_SO"
          '';
        };

        # nix develop .#trainer
        # runs trainer.py (wake word sample recorder) and sets up the
        # CoreWorxLab training pipeline in a local venv.
        #
        # NOTE: the CoreWorxLab docker container is CUDA-only — your RX 6900 XT
        # won't be used by it. to actually use ROCm, skip docker and run training
        # via the venv directly (pytorch-rocm is pip-installable):
        #   pip install torch torchvision torchaudio \
        #     --index-url https://download.pytorch.org/whl/rocm6.1
        # then run train.py directly instead of via docker compose.
        trainer = pkgs.mkShell {
        name = "wakeword-trainer";

        packages = [
          trainerPythonEnv
          pkgs.python3Packages.pip
          pkgs.portaudio   # pyaudio C dep
          pkgs.sox         # useful for inspecting recorded wavs
          pkgs.ffmpeg      # augmentation / format conversion
          pkgs.docker      # for the CoreWorxLab docker compose path
          pkgs.docker-compose
          pkgs.git
        ] ++ runtimeLibs;

        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimeLibs}:$LD_LIBRARY_PATH"

          if [ ! -d .venv-trainer ]; then
            echo "creating trainer venv..."
            python -m venv --system-site-packages .venv-trainer
            echo "installing openwakeword + training deps..."
            .venv-trainer/bin/pip install -q \
              openwakeword \
              onnxruntime \
              scipy \
              tqdm \
              pyyaml
            echo ""
            echo "  to use your RX 6900 XT for training (skips docker):"
            echo "  .venv-trainer/bin/pip install torch torchvision torchaudio \\"
            echo "    --index-url https://download.pytorch.org/whl/rocm6.1"
          fi
          source .venv-trainer/bin/activate

          # clone CoreWorxLab training repo if not present
          if [ ! -d openwakeword-training ]; then
            echo "cloning CoreWorxLab/openwakeword-training..."
            git clone --quiet https://github.com/CoreWorxLab/openwakeword-training.git
          fi

          echo ""
          echo "wakeword trainer shell ready"
          echo ""
          echo "  1. record samples:    python trainer.py"
          echo "     open http://<your-ip>:5002 on your phone"
          echo ""
          echo "  2a. train (docker/CPU, 6-8h per word):"
          echo "      cd openwakeword-training && docker compose build trainer"
          echo "      cp ../wakeword_samples/hey_zane/positive/*.wav data/hey_zane/positive/"
          echo "      cp ../wakeword_samples/zane_write/positive/*.wav data/zane_write/positive/"
          echo '      docker compose run --rm trainer python train.py --wake-word "hey zane"'
          echo '      docker compose run --rm trainer python train.py --wake-word "zane write"'
          echo ""
          echo "  2b. train (ROCm/GPU, faster — install torch-rocm first, see above):"
          echo "      cd openwakeword-training"
          echo '      python train.py --wake-word "hey zane"'
          echo '      python train.py --wake-word "zane write"'
          echo ""
          echo "  3. test: python test_model.py --model my_custom_model/hey_zane.onnx"
          echo ""
        '';
        };  # trainer
      };    # devShells.${system}
    };      # outputs
}
