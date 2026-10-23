"""Process-wide service state, in its own module so routers and the app can both reach it
without importing each other."""
STATE: dict = {}
