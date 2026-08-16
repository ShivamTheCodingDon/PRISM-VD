"""
N-Day CVE Database for Real-World Vulnerability Testing
========================================================
Hardcoded database of known CVEs with file path patterns and function names.
Used by extract_functions.py to auto-label functions as vulnerable (label=1).

Each entry contains:
  - cve_id:      CVE identifier
  - cwe:         CWE classification
  - project:     Source project name
  - file_pattern: Regex pattern matching the vulnerable file path
  - func_name:   Name of the vulnerable function
  - description: Brief description of the vulnerability
"""

NDAY_CVE_DATABASE = [
    # =========================================================================
    # LINUX KERNEL
    # =========================================================================
    {
        "cve_id": "CVE-2017-7308",
        "cwe": "CWE-119",
        "project": "linux",
        "file_pattern": r"net/packet/af_packet\.c",
        "func_name": "packet_set_ring",
        "description": "Heap out-of-bounds in packet_set_ring due to improper bounds check on tp_block_size"
    },
    {
        "cve_id": "CVE-2016-5195",
        "cwe": "CWE-362",
        "project": "linux",
        "file_pattern": r"mm/gup\.c",
        "func_name": "follow_page_pte",
        "description": "Dirty COW: race condition in get_user_pages allows privilege escalation"
    },
    {
        "cve_id": "CVE-2017-6074",
        "cwe": "CWE-416",
        "project": "linux",
        "file_pattern": r"net/dccp/input\.c",
        "func_name": "dccp_rcv_state_process",
        "description": "Use-after-free in DCCP protocol via DCCP_PKT_REQUEST"
    },
    {
        "cve_id": "CVE-2017-11176",
        "cwe": "CWE-416",
        "project": "linux",
        "file_pattern": r"ipc/mqueue\.c",
        "func_name": "mq_notify",
        "description": "Use-after-free in mq_notify via netlink socket"
    },
    {
        "cve_id": "CVE-2016-0728",
        "cwe": "CWE-416",
        "project": "linux",
        "file_pattern": r"security/keys/process_keys\.c",
        "func_name": "join_session_keyring",
        "description": "Use-after-free in keyring facility via keyctl"
    },
    {
        "cve_id": "CVE-2017-10661",
        "cwe": "CWE-362",
        "project": "linux",
        "file_pattern": r"fs/timerfd\.c",
        "func_name": "timerfd_setup_cancel",
        "description": "Race condition in timerfd allows read-after-free"
    },
    {
        "cve_id": "CVE-2017-1000112",
        "cwe": "CWE-362",
        "project": "linux",
        "file_pattern": r"net/ipv4/ip_output\.c",
        "func_name": "ip_do_fragment",
        "description": "Exploitable memory corruption via UFO to non-UFO path switch"
    },

    # =========================================================================
    # OPENSSL
    # =========================================================================
    {
        "cve_id": "CVE-2014-0160",
        "cwe": "CWE-119",
        "project": "openssl",
        "file_pattern": r"ssl/d1_both\.c|ssl/t1_lib\.c",
        "func_name": "dtls1_process_heartbeat",
        "description": "Heartbleed: buffer over-read in TLS heartbeat extension"
    },
    {
        "cve_id": "CVE-2014-0160",
        "cwe": "CWE-119",
        "project": "openssl",
        "file_pattern": r"ssl/t1_lib\.c",
        "func_name": "tls1_process_heartbeat",
        "description": "Heartbleed: buffer over-read in TLS heartbeat extension (TLS variant)"
    },
    {
        "cve_id": "CVE-2016-0799",
        "cwe": "CWE-119",
        "project": "openssl",
        "file_pattern": r"crypto/bio/b_print\.c",
        "func_name": "doapr_outch",
        "description": "Heap buffer overflow in BIO_*printf functions"
    },
    {
        "cve_id": "CVE-2014-3512",
        "cwe": "CWE-119",
        "project": "openssl",
        "file_pattern": r"ssl/s3_pkt\.c",
        "func_name": "ssl3_read_bytes",
        "description": "Buffer overflow via crafted DTLS fragment"
    },
    {
        "cve_id": "CVE-2016-2108",
        "cwe": "CWE-119",
        "project": "openssl",
        "file_pattern": r"crypto/asn1/a_d2i_fp\.c",
        "func_name": "asn1_d2i_read_bio",
        "description": "Memory corruption in ASN.1 encoder"
    },

    # =========================================================================
    # FFMPEG
    # =========================================================================
    {
        "cve_id": "CVE-2016-10190",
        "cwe": "CWE-119",
        "project": "ffmpeg",
        "file_pattern": r"libavformat/http\.c",
        "func_name": "http_read",
        "description": "Heap buffer overflow in HTTP protocol handler"
    },
    {
        "cve_id": "CVE-2016-10191",
        "cwe": "CWE-119",
        "project": "ffmpeg",
        "file_pattern": r"libavformat/rtmppkt\.c",
        "func_name": "ff_rtmp_packet_read",
        "description": "Heap buffer overflow in RTMP packet reader"
    },
    {
        "cve_id": "CVE-2017-9992",
        "cwe": "CWE-119",
        "project": "ffmpeg",
        "file_pattern": r"libavcodec/dfa\.c",
        "func_name": "decode_dds1",
        "description": "Heap buffer overflow in DFA video decoder"
    },
    {
        "cve_id": "CVE-2018-1999011",
        "cwe": "CWE-119",
        "project": "ffmpeg",
        "file_pattern": r"libavcodec/asfdec_f\.c|libavformat/asfdec_f\.c",
        "func_name": "asf_read_header",
        "description": "Heap buffer overflow in ASF demuxer"
    },

    # =========================================================================
    # QEMU
    # =========================================================================
    {
        "cve_id": "CVE-2015-3456",
        "cwe": "CWE-119",
        "project": "qemu",
        "file_pattern": r"hw/block/fdc\.c",
        "func_name": "fdctrl_handle_drive_specification_command",
        "description": "VENOM: buffer overflow in floppy disk controller"
    },
    {
        "cve_id": "CVE-2015-5158",
        "cwe": "CWE-476",
        "project": "qemu",
        "file_pattern": r"hw/scsi/scsi-bus\.c",
        "func_name": "scsi_req_complete",
        "description": "NULL pointer dereference in SCSI bus"
    },
    {
        "cve_id": "CVE-2020-14364",
        "cwe": "CWE-787",
        "project": "qemu",
        "file_pattern": r"hw/usb/core\.c",
        "func_name": "do_token_in",
        "description": "Out-of-bounds read/write in USB emulation"
    },

    # =========================================================================
    # XEN
    # =========================================================================
    {
        "cve_id": "CVE-2017-15595",
        "cwe": "CWE-835",
        "project": "xen",
        "file_pattern": r"xen/common/grant_table\.c",
        "func_name": "__gnttab_unmap_common_complete",
        "description": "Infinite loop in grant table operations"
    },

    # =========================================================================
    # LIBAV
    # =========================================================================
    {
        "cve_id": "CVE-2015-3417",
        "cwe": "CWE-119",
        "project": "libav",
        "file_pattern": r"libavcodec/h264_cabac\.c",
        "func_name": "decode_cabac_residual_nondc",
        "description": "Buffer overflow in H.264 CABAC decoder"
    },
]


def get_cves_for_project(project: str) -> list:
    """Return CVE entries for a given project name."""
    proj = project.lower()
    return [cve for cve in NDAY_CVE_DATABASE if cve["project"] == proj]


def get_vulnerable_function_set(project: str) -> set:
    """Return a set of (file_pattern, func_name) tuples for a project."""
    return {(cve["file_pattern"], cve["func_name"]) for cve in get_cves_for_project(project)}


def is_function_vulnerable(project: str, file_path: str, func_name: str) -> dict | None:
    """
    Check if a function matches any known CVE.
    Returns the CVE entry dict if matched, else None.
    """
    import re
    for cve in get_cves_for_project(project):
        if cve["func_name"] == func_name:
            if re.search(cve["file_pattern"], file_path):
                return cve
    return None


if __name__ == "__main__":
    print(f"Total CVEs in database: {len(NDAY_CVE_DATABASE)}")
    projects = set(c["project"] for c in NDAY_CVE_DATABASE)
    for proj in sorted(projects):
        cves = get_cves_for_project(proj)
        print(f"  {proj}: {len(cves)} CVEs")
        for c in cves:
            print(f"    - {c['cve_id']}: {c['func_name']} ({c['cwe']})")
