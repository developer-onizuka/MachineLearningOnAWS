# -*- mode: ruby -*-
# vi: set ft=ruby :
# virsh pool-create-as --name data --type dir --target /mnt/data

Vagrant.configure("2") do |config|
  config.vm.box = "generic/ubuntu2004"

  config.vm.define "ubuntu_192.168.33.119"
  config.vm.network "private_network", ip: "192.168.33.119"
  config.vm.network "private_network", ip: "192.168.133.119"
  config.vm.hostname = "tensorflow"

  config.vm.provider "libvirt" do |kvm|
    kvm.storage_pool_name = "data"
    # kvm.memory = 49152
    kvm.memory = 36864 
    kvm.cpus = 4
    kvm.machine_type = "q35"
    kvm.cpu_mode = "host-passthrough"
    kvm.kvm_hidden = true
    kvm.pci :bus => '0x01', :slot => '0x00', :function => '0x0'
  end

  config.vm.provision "shell", inline: <<-SHELL
     apt-get update
     apt-get install -y docker.io
     apt-get install -y curl
     curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
     distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
     curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list |sudo tee /etc/apt/sources.list.d/nvidia-docker.list
     apt-get update
     apt-get install -y nvidia-docker2
     systemctl restart docker
     cat <<EOF > /etc/docker/daemon.json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "insecure-registries":["192.168.133.1:5000"]
}
EOF
     sed -i 's/^#root/root/' /etc/nvidia-container-runtime/config.toml
     tee /etc/modules-load.d/ipmi.conf <<< "ipmi_msghandler"   && sudo tee /etc/modprobe.d/blacklist-nouveau.conf <<< "blacklist nouveau"   && sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf <<< "options nouveau modeset=0"
     update-initramfs -u
     docker pull nvcr.io/nvidia/driver:535.129.03-ubuntu20.04
     docker pull nvcr.io/nvidia/tensorflow:23.07-tf2-py3
   SHELL
end
