#[tokio::main]
async fn main() -> anyhow::Result<()> {
    mnemosyne_api::run().await
}
