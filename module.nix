{pkgs, lib, ...}:
let
  libfreenect2 = pkgs.stdenv.mkDerivation rec {
      pname = "libfreenect2";
      version = "0.2.0";

      src = pkgs.fetchFromGitHub {
        owner = "OpenKinect";
        repo = "libfreenect2";
        rev = "v${version}";
        hash = "sha256-5JjZANfkkgK8YRXrdfOCXOBMXxW+UNv6JiNfEDUQscc=";
      };

      nativeBuildInputs = with pkgs; [
        cmake
        libusb1
        pkg-config
      ];

      buildInputs = with pkgs; [
        pkg-config
        libusb1
        libusb1
        glfw3
        libjpeg_turbo
        libGL
        libGLU
        opencl-headers
        ocl-icd
      ];

      cmakeFlags = [
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.10"
        "-DBUILD_EXAMPLES=ON"
        "-DENABLE_OPENGL=ON"
        "-DENABLE_OPENCL=OFF"
        "-DENABLE_CUDA=OFF"
      ];

      postInstall = ''
        mkdir -p $out/lib/udev/rules.d
        cp ../platform/linux/udev/90-kinect2.rules $out/lib/udev/rules.d/90-kinect2.rules
      '';
    };

    in
{
  services.udev.extraRules = ''
      # Xbox NUI Sensor (Kinect v2)
      SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02c4", MODE="0666", GROUP="video"
      SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02d8", MODE="0666", GROUP="video"
      SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02d9", MODE="0666", GROUP="video"
    '';

    environment.systemPackages = [
      libfreenect2
    ];

}
