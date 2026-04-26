---
title: "Ephemeral Playground"
tags: ["microvm", "firecracker"]
ShowToc: true
---

Ever wanted a throw-away VM that spins quickly? One good example of this use case is practice playground VM on learning sites.

In this project, we will create a CLI that provisions a throw-away VM for us. It uses [firecracker](https://github.com/firecracker-microvm/firecracker) microVM underneath to provision and destroy the VMs.

## Getting Started

Firecracker has pretty good [getting started](https://github.com/firecracker-microvm/firecracker/blob/main/docs/getting-started.md) guide. I will use it as a reference to get started. 

First step is to look if `/dev/kvm` is available on the host machine. This is the important requirement to use firecracker. The presence of KVM module can be checked using `lsmod | grep kvm` command. The following command checks if you have read and write access to `/dev/kvm`.

```bash
[ -r /dev/kvm ] && [ -w /dev/kvm ] && echo "OK" || echo "FAIL"
```

### Running Firecracker

Following steps are required to start a VM using Firecracker

#### Start firecracker with an API socket

Download the latest release of firecracker binary from the [release](https://github.com/firecracker-microvm/firecracker/releases) page for your platform. Firecracker supports **x86_64** and **aarch64** Linux platforms. I am using [`firecracker-v1.15.1-x86_64.tgz`](https://github.com/firecracker-microvm/firecracker/releases/download/v1.15.1/firecracker-v1.15.1-x86_64.tgz). The alternate approach is to build firecracker from source.

```bash
curl -OL "https://github.com/firecracker-microvm/firecracker/releases/download/v1.15.1/firecracker-v1.15.1-x86_64.tgz"
curl -OL "https://github.com/firecracker-microvm/firecracker/releases/download/v1.15.1/firecracker-v1.15.1-x86_64.tgz.sha256.txt"
sha256sum --check firecracker-v1.15.1-x86_64.tgz.sha256.txt
tar -xvzf firecracker-v1.15.1-x86_64.tgz
# Rename the firecracker version binary to firecracker
mv release-v1.15.1-x86_64/firecracker-v1.15.1-x86_64 firecracker
```

Starting the firecracker on a separate terminal. This creates a socket file at `/tmp/firecracker.socket` that we will use to send API requests to firecracker. The following command might require `sudo` if your current user does not have read and write access to `/dev/kvm`.

```bash
API_SOCKET="/tmp/firecracker.socket"
# Remove if exists to avoid `AddrInUse` error
rm -f $API_SOCKET
# Run firecracker
./firecracker --api-sock "${API_SOCKET}" --enable-pci
```

### Configure bootsource

To get started, we will use latest release-matched guest kernel artifact published by Firecracker CI. The kernel is downloaded directly using following commands. The contents of the bucket can be viewed using `aws s3 ls s3://spec.ccfc.min/firecracker-ci/v1.15/x86_64/ --no-sign-request` command.

```bash
mkdir -p artifacts
CI_VERSION="v1.15"
ARCH="x86_64"
IMAGE_BASE_URL="https://s3.amazonaws.com/spec.ccfc.min/firecracker-ci/${CI_VERSION}/${ARCH}"
KERNEL_URL="${IMAGE_BASE_URL}/vmlinux-6.1.155"
curl -fsSL -o ./artifacts/vmlinux.bin "${KERNEL_URL}"
```

The `vmlinux.bin` contains the guest kernel image which is under `artifacts` folder locally. The `strings` command can be used to inspect the contents of the kernel. 

```bash 
cd artifacts
strings vmlinux.bin | grep -m1 '^Linux version '

Linux version 6.1.155+ (root@c3d8009cd6d2) (gcc (Ubuntu 11.4.0-1ubuntu1~22.04.2) 11.4.0, GNU ld (GNU Binutils for Ubuntu) 2.38) # SMP PREEMPT_DYNAMIC
```

### Configure rootfs

The Firecracker CI is again used to download the root file system (rootfs) artifact. The Ubuntu rootfs is published as a `squashfs` image and converted locally into an `ext4` root filesystem image before booting.

```bash
CI_VERSION="v1.15"
ARCH="x86_64"
IMAGE_BASE_URL="https://s3.amazonaws.com/spec.ccfc.min/firecracker-ci/${CI_VERSION}/${ARCH}"
ROOTFS_URL="${IMAGE_BASE_URL}/ubuntu-24.04.squashfs"
curl -fsSL -o ./artifacts/ubuntu.squashfs "${ROOTFS_URL}"
```

Create a `ext4` from `squashfs` image.

```bash
cd artifacts
unsquashfs ubuntu.squashfs
sudo chown -R root:root squashfs-root
truncate -s 1G rootfs.ext4
sudo mkfs.ext4 -d squashfs-root -F rootfs.ext4
```

The `rootfs.ext4` is the guest root filesystem image saved under `artifacts` folder locally. This will be mounted as `/` the root filesystem. The following shows the tree structure of root file system and the OS information.

```bash
cd artifacts
debugfs -R 'ls -l /' rootfs.ext4
debugfs -R 'cat /usr/lib/os-release' rootfs.ext4

PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.3 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
...
```

Both the rootfs and kernel are using `Ubuntu 24.04.3 LTS` with Linux kernel version `6.1`. Firecracker docs also provide guide on [creating custom rootfs and kernel images](https://github.com/firecracker-microvm/firecracker/blob/main/docs/rootfs-and-kernel-setup.md).

### Start the microVM

Now we are all set to boot our first VM. Let's walk through set by step on what is required to start the microVM starting with fixing few variables for the artifacts and sockets configured in the previous step.

```bash
API_SOCKET="/tmp/firecracker.socket"
KERNEL="$PWD/artifacts/vmlinux.bin"
ROOTFS="$PWD/artifacts/rootfs.ext4"
BOOT_ARGS='console=ttyS0 reboot=k panic=1'
```

Next step is to send API request to firecracker to configure the machine size. Here, I set 1 vCPU and 256 MB of memory for the microVM. 

```bash 
curl --unix-socket "$API_SOCKET" -i \
-X PUT 'http://localhost/machine-config' \
-H 'Content-Type: application/json' \
-d '{
    "vcpu_count": 1,
    "mem_size_mib": 256,
    "smt": false
}'
```

The response for this `PUT` request would be `HTTP/1.1 204` No content. 

Followed by configuring the boot source and root filesystem requests to the API.

```bash
# Boot source
curl --unix-socket "$API_SOCKET" -i \
-X PUT 'http://localhost/boot-source' \
-H 'Content-Type: application/json' \
-d "{
  \"kernel_image_path\": \"$KERNEL\",
  \"boot_args\": \"$BOOT_ARGS\"
}"

# Root filesystem
curl --unix-socket "$API_SOCKET" -i \
-X PUT 'http://localhost/drives/rootfs' \
-H 'Content-Type: application/json' \
-d "{
  \"drive_id\": \"rootfs\",
  \"path_on_host\": \"$ROOTFS\",
  \"is_root_device\": true,
  \"is_read_only\": false
}"
```

The `InstanceStart` action powers on the microVM and starts the guest OS.

```bash
# The docs insert a tiny wait before InstanceStart
sleep 0.05

# Start the microVM
curl --unix-socket "$API_SOCKET" -i \
-X PUT 'http://localhost/actions' \
-H 'Content-Type: application/json' \
-d '{
    "action_type": "InstanceStart"
}'
``` 
   
Since we enabled the serial console in the `BOOT_ARGS`, the guest serial output goes to the Firecracker's stdout. Once the everything above is a success, we are dropped into the ttyS0 console of the microVM and see the shell.

```bash
Ubuntu 24.04.3 LTS ubuntu-fc-uvm ttyS0

ubuntu-fc-uvm login: root (automatic login)

Welcome to Ubuntu 24.04.3 LTS (GNU/Linux 6.1.155+ x86_64)

* Documentation:  https://help.ubuntu.com
* Management:     https://landscape.canonical.com
* Support:        https://ubuntu.com/pro

This system has been minimized by removing packages and content that are
not required on a system that users do not log into.

To restore this content, you can run the 'unminimize' command.

The programs included with the Ubuntu system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Ubuntu comes with ABSOLUTELY NO WARRANTY, to the extent permitted by
applicable law.

root@ubuntu-fc-uvm:~# uname -a
Linux ubuntu-fc-uvm 6.1.155+ #1 SMP PREEMPT_DYNAMIC Thu Dec 18 15:17:16 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux
root@ubuntu-fc-uvm:~# nproc
1
root@ubuntu-fc-uvm:~# free
                total        used        free      shared  buff/cache   available
Mem:          236088       32720      177184        1636       36016      203368
Swap:              0           0           0
```

> [!NOTE]
> There are several other approaches supported in addition to sending API request such as using a configuration file or SDKs in different programming languages. The API approach is used here for demonstration purposes and to have more control over the steps involved in provisioning the microVM.

### Terminate microVM

To terminate the microVM, run the `reboot` command in the shell.

```bash
[  402.496207] reboot: Restarting system
[anonymous-instance:main] Vmm is stopping.
[anonymous-instance:main] Killing vCPU threads
[anonymous-instance:main] Firecracker exiting successfully. exit_code=0
```

## Ephemeral Playground

Now that we have successfully provisioned and deprovisioned our first microVM, we can wrap the above steps into a CLI that provisions a throw-away VM for us. The CLI will be built using [`just`](https://just.systems/man/en/introduction.html) task runner.

> [!CODE]
> Github: https://github.com/dudeperf3ct/ephemeral-playground

All the above process can be boiled down to running two commands in two separate terminals. In one terminal, start the firecracker with an API socket

```bash
just start-firecracker-console
```

In second terminal, configure the machine used for microVM along with kernel and rootfs for guest OS.

```bash
just boot-vm
```

Once the microVM is booted, you can run the commands in the console on the mircoVM in the first terminal. 

### Snapshotting and Restoring

One of the cool features of the microVM is you can create multiple identical copies of same microVM. As part of snapshotting process, firecracker saves current microVM state and memory. There are two ways to create a snapshot: full or diff. The diff version is in developer preview as of April 2026. The diff version saves a diff of current microVM state since the last snapshot. Snapshotting helps with the cold time starts of the microVMs.

If we look into the [code](https://github.com/firecracker-microvm/firecracker/blob/2440dbb2354dc5a5734e67f7f429530cdf2f8714/src/vmm/src/arch/x86_64/vm.rs#L234) for `x86_64` architecture, `VmState` struct holds the guest memory state which consists of the list of guest memory regions and their snapshot file mappings. 

```Rust
/// Structure holding VM kvm state.
pub struct VmState {
    /// guest memory state
    pub memory: GuestMemoryState,
    /// resource allocator
    pub resource_allocator: ResourceAllocator,
    pitstate: kvm_pit_state2,
    clock: kvm_clock_data,
    // TODO: rename this field to adopt inclusive language once Linux updates it, too.
    pic_master: kvm_irqchip,
    // TODO: rename this field to adopt inclusive language once Linux updates it, too.
    pic_slave: kvm_irqchip,
    ioapic: kvm_irqchip,
}

/// State of a guest memory region saved to file/buffer.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestMemoryRegionState {
    // This should have been named `base_guest_addr` since it's _guest_ addr, but for
    // backward compatibility we have to keep this name. At least this comment should help.
    /// Base GuestAddress.
    pub base_address: u64,
    /// Region size.
    pub size: usize,
    /// Region type
    pub region_type: GuestRegionType,
    /// Plugged/unplugged status of each slot
    pub plugged: Vec<bool>,
}

/// Describes guest memory regions and their snapshot file mappings.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestMemoryState {
    /// List of regions.
    pub regions: Vec<GuestMemoryRegionState>,
}
```

Restoring the snapshot allows one to restore and resume from the full snapshotted microVM state. The loading for diff version is bit different as it requires merging the diff states with base states to create similar to rebasing concept in git. The order in which the snapshots were created matters and they should be merged in the same order in which they were created. Care must be taken for [random numbers](https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/random-for-clones.md) when restoring multiple VM clones from single snapshot.

In our case, we can boot a warm playground fast if we create multiple copies of base with all dependencies required for the project.

```bash
just snapshot-create latest Full
```

`snapshot-create` command saves the metadata, memory file and current microVM state as artifacts.

```bash
just snapshot-restore latest true
```

`snapshot-restore` command restores and resumes the microVM from snapshot artifact.

### Networking

If we run any `apt-get install` or `curl` commands that require internet access, the microVM returns the familiar `could not resolve` error. Firecracker documentation provides a [getting started](https://github.com/firecracker-microvm/firecracker/blob/main/docs/network-setup.md#getting-started-firecracker-network-setup) guide for networking setup.

Before wiring networking into the microVM, let's look at what happens on a normal Linux system when we run `curl example.com`. I am simplifying some of the DNS lookup and caching details here.

1. The application first needs to resolve the domain name `example.com` to an IP address. It asks the system resolver, which uses DNS configuration such as the nameservers listed in `/etc/resolv.conf`.

2. If the answer is not already known locally, the resolver may need to contact a DNS server. This DNS lookup is itself a network request.

3. The kernel then uses the routing table to decide which network interface should carry that packet. That interface could be a physical interface such as Ethernet (`eth0`) or Wi-Fi (`wlan0`), or a virtual interface such as loopback (`lo`).

4. Once the outgoing interface is chosen, the packet is transmitted through the network interface card (NIC), which is responsible for sending and receiving network traffic.

Inside a Firecracker microVM, the first step is basically the same. When a process in the guest makes a network request, the guest first uses its own system resolver and `/etc/resolv.conf` to translate the domain name into an IP address.

The difference starts after that. On a normal host, the kernel may send the packet through a real NIC. Inside the guest, the kernel still builds the packet and chooses an outgoing interface using its routing table, but that interface is not backed by a physical NIC. Instead, it is a virtual network interface backed by `virtio-net`. You can think of `virtio-net` as the microVM's virtual network card.

Firecracker connects this virtual interface to a host-side TAP device. A TAP device is a virtual Ethernet interface on the host that acts like the other end of the guest's virtual NIC. From there, the host kernel can forward the packet onward through the host's real network interface to the outside network. 

Firecracker's basic networking model is: guest `virtio-net` -> host TAP device -> host routing/NAT -> internet. Let's use this model to build what is required on both the host and guest side.

#### Create a virtual cable on the host

  A Linux TAP device is a software Ethernet interface. Firecracker uses it as the host-side end of the guest's virtual network card. We create `tap0`, give it an IP like `172.16.0.1/30`, and bring it up. We use a `/30` subnet here because we only need two working IP addresses: one for the TAP device and one for the guest.

```bash
sudo ip tuntap add tap0 mode tap
sudo ip addr add 172.16.0.1/30 dev tap0
sudo ip link set tap0 up
```

#### Let the host forward packets

The guest cannot reach the internet unless the host forwards packets on its behalf. We enable IPv4 forwarding and add NAT rules so packets from the guest are masqueraded as if they came from the host's real network interface.

Firecracker's guide shows two ways to do this: `nft`, which is the recommended modern approach, and `iptables-nft`, which is included mainly for compatibility and familiarity. The commands differ, but the result is the same: the host forwards packets from the guest to the internet and back.

```bash
echo 1 | sudo tee /proc/sys/net/ipv4/ip_forward
sudo iptables-nft -t nat -A POSTROUTING -o eth0 -s 172.16.0.2 -j MASQUERADE
sudo iptables-nft -A FORWARD -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
sudo iptables-nft -A FORWARD -i tap0 -o eth0 -j ACCEPT
```

#### Attach the VM NIC to that TAP

Now we tell Firecracker: "this microVM should have one network interface, and its host backing device is `tap0`." That is done through the `network-interfaces` API endpoint.

```bash
curl --unix-socket /tmp/firecracker.socket -i \
  -X PUT 'http://localhost/network-interfaces/my_network0' \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
      "iface_id": "my_network0",
      "guest_mac": "06:00:AC:10:00:02",
      "host_dev_name": "tap0"
    }'
```

#### Configure the guest

Inside the guest, `eth0` needs an IP address, a default gateway, and DNS. We can either configure this manually after boot or pass Linux the `ip=` kernel argument so it does part of the setup automatically.

```bash
ip addr add 172.16.0.2/30 dev eth0
ip link set eth0 up
ip route add default via 172.16.0.1 dev eth0
printf 'nameserver 8.8.8.8\n' > /etc/resolv.conf
```

#### Test

```bash
ping -c 1 172.16.0.1 # guest can reach host TAP
ping -c 1 1.1.1.1  # guest has raw internet routing
curl https://example.com  # routing + DNS both work
```

> [!NOTE]
> The Firecracker's [networking guide](https://github.com/firecracker-microvm/firecracker/blob/main/docs/network-setup.md) mentions 3 approaches for configuring routing for a microVM: `NAT-based`, `Bridge-based` and `Namespaced-NAT`. We looked at only the NAT-based approach above. The networking also does not work out of the box for [clones](https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/network-for-clones.md) created from the same snapshots as of version 1.15.1.

All the above process is abstracted under `just network-up` and `just network-down` commands for our ephemeral playground. By default, `just boot-vm` takes care of setting up the host and guest networking. To start the microVM, open two terminals.

One the first terminal, run the following to start Firecracker API console,

```bash
just start-firecracker-console
```

On the second terminal, boot the microVM using

```bash
just boot-vm
```

The first terminal should give a shell the microVM. Run the following commands in the console to check the network connectivity.

```bash
ping -c 5 172.16.0.1 # guest can reach host TAP
ping -c 5 1.1.1.1  # guest has raw internet routing
curl https://example.com  # routing + DNS both work
```

To stop the firecracker and exit the microVM, run `just stop-firecracker` command in the second terminal.

> [!EXAMPLE]
> I came across the blog [The invisible engineering behind Lambda’s network](https://www.allthingsdistributed.com/2026/04/the-invisible-engineering-behind-lambdas-network.html) that talks about different challenges of using microVM for AWS Lambda at scale. The difficulties with snapshotting, networking bottlenecks detailing all the invisible engineering that takes place to make Lambda functions start faster and run efficiently.