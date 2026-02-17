from tusas import create_app


app = create_app()


if __name__ == "__main__":
    import webbrowser
    import threading
    def open_browser():
        import time
        time.sleep(1.2)
        webbrowser.open("http://127.0.0.1:5000/")
    print("\n  Tarayicida acin: http://127.0.0.1:5000")
    print("  (https DEGIL, http kullanin - sunucu acik kalmali.)\n")
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)
