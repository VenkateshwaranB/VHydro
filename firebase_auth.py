import streamlit as st
import re
import time
from datetime import datetime

# Hardcoded users for demonstration (replace with proper authentication later)
DEMO_USERS = {
    "user@example.com": "Password123",
    "admin@vhydro.com": "Admin123"
}

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
    
    if email in DEMO_USERS:
        return False, "Email already in use"
    
    # Add user to demo users (in a real app, you would save to a database)
    DEMO_USERS[email] = password
    
    # Store user info in session state
    st.session_state["user_id"] = email
    st.session_state["email"] = email
    st.session_state["logged_in"] = True
    
    return True, "Account created successfully!"

# Function to handle login
def login(email, password):
    if not email or not password:
        return False, "Please enter both email and password"
    
    # Check if user exists and password matches
    if email in DEMO_USERS and DEMO_USERS[email] == password:
        # Store user info in session state
        st.session_state["user_id"] = email
        st.session_state["email"] = email
        st.session_state["logged_in"] = True
        
        return True, "Login successful!"
    else:
        return False, "Invalid email or password"

# Function to handle password reset (simplified mock version)
def reset_password(email):
    if not is_valid_email(email):
        return False, "Invalid email format"
    
    if email in DEMO_USERS:
        return True, "Password reset email would be sent! (Demo only)"
    else:
        return False, "Email not found"

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
    .auth-header {
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add title and description
    st.markdown("<h1 style='text-align: center; color: #0c326f;'>VHydro Login</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Hydrocarbon Quality Prediction</p>", unsafe_allow_html=True)
    
    # Create tabs for Login and Sign Up
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    # Login tab
    with tab1:
        st.markdown('<div class="auth-form">', unsafe_allow_html=True)
        st.markdown('<h2 class="auth-header">Login</h2>', unsafe_allow_html=True)
        
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            login_button = st.button("Login", use_container_width=True)
        
        if login_button:
            success, message = login(email, password)
            if success:
                st.success(message)
                time.sleep(1)  # Wait for a second to show success message
                st.rerun()
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
        st.markdown('<h2 class="auth-header">Create Account</h2>', unsafe_allow_html=True)
        
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        # Password requirements notice
        st.markdown("""
        <div style="font-size: 0.8em; color: #6c757d; margin-bottom: 15px;">
        Password must be at least 8 characters with 1 uppercase letter, 1 lowercase letter, and 1 digit.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            signup_button = st.button("Sign Up", use_container_width=True)
        
        if signup_button:
            success, message = signup(new_email, new_password, confirm_password)
            if success:
                st.success(message)
                time.sleep(1)  # Wait for a second to show success message
                st.rerun()
            else:
                st.error(message)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return False

# Function to show user profile/account page
def user_account_page():
    st.markdown("<h2 class='sub-header'>My Account</h2>", unsafe_allow_html=True)
    
    if not st.session_state.get("logged_in", False):
        st.warning("Your session has expired. Please log in again.")
        logout()
        st.rerun()
        return
    
    # Create a card-like interface for user info
    st.markdown("""
    <style>
    .user-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-header {
        border-bottom: 1px solid #e9ecef;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    .user-field {
        margin-bottom: 15px;
    }
    .field-label {
        font-weight: bold;
        color: #0e4194;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="user-card">', unsafe_allow_html=True)
    st.markdown('<div class="user-header">', unsafe_allow_html=True)
    st.markdown(f"<h3>👤 {st.session_state.get('email', 'User')}</h3>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="user-field">', unsafe_allow_html=True)
    st.markdown('<span class="field-label">Email:</span>', unsafe_allow_html=True)
    st.write(st.session_state.get('email', 'Unknown'))
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Logout section
    st.markdown("<h3 class='section-header'>Logout</h3>", unsafe_allow_html=True)
    
    st.markdown('<div class="user-card">', unsafe_allow_html=True)
    st.write("Click the button below to log out of your account.")
    
    if st.button("Logout"):
        logout()
        st.success("You have been logged out!")
        time.sleep(1)
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
