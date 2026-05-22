import os, pathlib
cred = pathlib.Path(os.environ["APPDATA"]) / "streamlit" / "credentials.toml"
cred.parent.mkdir(parents=True, exist_ok=True)
cred.write_text('[general]\nemail = ""\n')
print("Credentials written to:", cred)
