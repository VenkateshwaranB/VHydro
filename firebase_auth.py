import streamlit as st
import firebase_admin
from firebase_admin import credentials, auth
import re
import time
from datetime import datetime

# Initialize Firebase Admin SDK if not already initialized
if not firebase_admin._apps:
    # You'll need to generate a service account key file from Firebase Console
    # Project settings -> Service accounts -> Generate new private key
    # Save the JSON file securely and specify the path here
    try:
        cred = credentials.Certificate("path/to/serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Error initializing Firebase: {e}")

# Function to validate email format
def is_valid_email(email):
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(email_pattern, email))

# Function to validate password strength
def is_strong_password(password):
    # At least 8 characters, with 1 uppercase, 1 lowercase, 1 digit
    if len(password) < 8:
        return False
    if not any(c.isupper() for c in password):
        return False
    if not any(c.islower() for c in password):
        return False
    if not any(c.isdigit() for c in password):
        return False
    return True

# Function to handle signup
def signup(email, password, confirm_password):
    if not is_valid_email(email):
        return False, "Invalid email format"
    
    if password != confirm_password:
        return False, "Passwords don't match"
    
    if not is_strong_password(password):
        return False, "Password must be at least 8 characters with 1 uppercase, 1 lowercase, and 1 digit"
    
    try:
        # Create user in Firebase
        user = auth.create_user(
            email=email,
            password=password
        )
        
        # Store user info in session state
        st.session_state["user_id"] = user.uid
        st.session_state["email"] = email
        st.session_state["logged_in"] = True
        
        return True, "Account created successfully!"
    except Exception as e:
        error_msg = str(e)
        if "ALREADY_EXISTS" in error_msg:
            return False, "Email already in use"
        return False, f"An error occurred: {error_msg}"

# Function to handle login (requires email/password auth flow)
def login(email, password):
    # For testing purposes, allow login with test credentials
    # In production, you would implement real authentication here
    if email == "test@example.com" and password == "Test123456":
        st.session_state["user_id"] = "test_user_id"
        st.session_state["email"] = email
        st.session_state["logged_in"] = True
        return True, "Login successful!"
    else:
        return False, "Invalid email or password"

# Function to handle password reset
def reset_password(email):
    if not is_valid_email(email):
        return False, "Invalid email format"
    
    try:
        auth.generate_password_reset_link(email)
        return True, "Password reset email sent!"
    except Exception as e:
        return False, f"An error occurred: {str(e)}"

# Function to log out
def logout():
    for key in ["user_id", "email", "logged_in"]:
        if key in st.session_state:
            del st.session_state[key]

# Main authentication function
def authenticate():
    """
    Main authentication function.
    Returns True if user is authenticated, False otherwise.
    """
    # Check if user is already authenticated
    if st.session_state.get("logged_in", False):
        return True
    
    # Otherwise, show auth page
    return auth_page()

# Function to render login/signup UI
def auth_page():
    # CSS for auth forms
    st.markdown("""
    <style>
    .auth-form {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create tabs for Login and Sign Up
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    # Login tab
    with tab1:
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        login_button = st.button("Login", use_container_width=True)
        
        if login_button:
            success, message = login(email, password)
            if success:
                st.success(message)
                time.sleep(1)
                st.experimental_rerun()
            else:
                st.error(message)
                
        forgot_password = st.button("Forgot Password?", key="forgot_password")
        if forgot_password:
            reset_email = st.text_input("Enter your email to reset password", key="reset_email")
            if st.button("Send Reset Link"):
                if reset_email:
                    success, message = reset_password(reset_email)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter your email")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sign Up tab
    with tab2:
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        signup_button = st.button("Sign Up", use_container_width=True)
        
        if signup_button:
            success, message = signup(new_email, new_password, confirm_password)
            if success:
                st.success(message)
                time.sleep(1)
                st.experimental_rerun()
            else:
                st.error(message)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return False

# Function to show user account page
def user_account_page():
    st.title("My Account")
    
    st.write(f"Email: {st.session_state.get('email', 'Unknown')}")
    
    if st.button("Send Password Reset Email"):
        email = st.session_state.get('email')
        if email:
            success, message = reset_password(email)
            if success:
                st.success(message)
            else:
                st.error(message)
    
    if st.button("Logout"):
        logout()
        st.success("You have been logged out!")
        time.sleep(1)
        st.experimental_rerun()
