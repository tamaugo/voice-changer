fn main() {
    println!("cargo:rerun-if-changed=../../assets/lark.ico");
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target == "windows" {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../../assets/lark.ico");
        res.compile().expect("embed Lark icon");
    }
}
